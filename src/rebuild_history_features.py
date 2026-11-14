from env import *
import pyogrio,shapely,gc,math
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.ops import transform
def run():
    keys=np.load(OLD/'source_key.npy',mmap_mode='r');order=np.load(OLD/'source_key_order.npy',mmap_mode='r');sk=keys[order]
    pc=(keys>>np.uint64(32)).astype(np.uint8);x=np.load(W/'cache/BAN_X.npy',mmap_mode='r');y=np.load(W/'cache/BAN_Y.npy',mmap_mode='r')
    label=np.load(W/'cache/analysis_first_year.npy');oldlabel=np.nan_to_num(np.load(OLD/'SHOUFA_YEA.npy',mmap_mode='r'),nan=0).astype(int)
    area=np.load(OLD/'Area.npy',mmap_mode='r');n=len(keys)
    positive=np.flatnonzero((label>=2000)&(label<2020)&np.isfinite(area)&(area>0))
    out={yy:np.lib.format.open_memmap(W/'cache'/f'inf_press_3km_{yy}.npy',mode='w+',dtype=np.float64,shape=(n,)) for yy in range(2001,2021)}
    for province in np.unique(pc):
        info=json.loads((W/'cache'/f'T{province}_info.json').read_text(encoding='utf-8'));bounds=info['total_bounds']
        lon0=(bounds[0]+bounds[2])/2;lat0=(bounds[1]+bounds[3])/2
        tr=Transformer.from_crs(4326,f'+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m',always_xy=True)
        target=np.flatnonzero(pc==province)
        # All case-source points close to the partition, including cross-partition sources.
        pad=.1;close=(x[positive]>=bounds[0]-pad)&(x[positive]<=bounds[2]+pad)&(y[positive]>=bounds[1]-pad)&(y[positive]<=bounds[3]+pad)
        src=positive[close];sx,sy=tr.transform(x[src],y[src]);tree=cKDTree(np.column_stack([sx,sy]))
        for at in range(0,len(target),30000):
            jj=target[at:at+30000];tx,ty=tr.transform(x[jj],y[jj]);tt=cKDTree(np.column_stack([tx,ty]))
            coo=tt.sparse_distance_matrix(tree,3000,output_type='coo_matrix');keep=jj[coo.row]!=src[coo.col]
            rr=coo.row[keep];ss=src[coo.col[keep]];dd=coo.data[keep]
            bins=rr*20+(label[ss]-2000)
            v=np.asarray(area[ss])*np.exp(-dd/1000)
            onset=np.bincount(bins,weights=v,minlength=len(jj)*20).reshape(len(jj),20)
            pressure=np.zeros(len(jj))
            for yy in range(2001,2021):
                pressure=math.exp(-1)*(pressure+onset[:,yy-2001])
                out[yy][jj]=pressure
        log(f'T{province}: recomputed annual pressure for {len(target):,} units')
    for a in out.values():a.flush()
    js(W/'cache/pressure_complete.json',dict(units=n,sources=len(positive),formula='sum Area_j exp(-d/1000) exp(-(y-t_j)) for d<=3000,t_j<y,i!=j',coordinates='inventory BAN_X/BAN_Y reference points; local AEQD; differs from original polygon centroids and must be disclosed',source='repaired first recorded year; source units outside main eligible cohort may still contribute recorded history'))
    log('Pressure rebuild completed; starting shared-boundary label correction')
    if '--pressure-only' in sys.argv:return
    boundary={yy:np.lib.format.open_memmap(W/'cache'/f'inf_by_{yy}.npy',mode='w+',dtype=np.float64,shape=(n,)) for yy in range(2001,2021)}
    for yy,a in boundary.items():a[:]=np.load(OLD/f'inf_by_{yy}.npy',mmap_mode='r')
    infos={int(p.stem[1:-5]):json.loads(p.read_text(encoding='utf-8')) for p in (W/'cache').glob('T*_info.json') if p.stem[1:-5].isdigit()}
    changes=np.flatnonzero(label!=oldlabel);edgecounts=[]
    for sourceprovince in np.unique(pc[changes]):
        rows=changes[pc[changes]==sourceprovince];oid=(keys[rows]&np.uint64(0xffffffff)).astype(np.int64)
        m,fi,gg,aa=pyogrio.raw.read(str(F/'Sub_Pine_All.gdb'),layer=f'T{sourceprovince}',columns=[],fids=oid.tolist(),return_fids=True)
        sourcegeoms={int(fid):g for fid,g in zip(fi,gg)};count=0
        info=infos[int(sourceprovince)];bb=info['total_bounds'];tr=Transformer.from_crs(4326,f'+proj=aeqd +lat_0={(bb[1]+bb[3])/2} +lon_0={(bb[0]+bb[2])/2} +datum=WGS84 +units=m',always_xy=True)
        for it,(row,o) in enumerate(zip(rows,oid)):
            sg=shapely.make_valid(shapely.from_wkb(sourcegeoms[int(o)]))
            if sg.is_empty:continue
            sb=sg.bounds;sm=shapely.set_precision(transform(tr.transform,sg),1.0)
            for tp,ti in infos.items():
                b=ti['total_bounds']
                if b[2]<sb[0]-.00002 or b[0]>sb[2]+.00002 or b[3]<sb[1]-.00002 or b[1]>sb[3]+.00002:continue
                mm,tf,tg,ta=pyogrio.raw.read(str(F/'Sub_Pine_All.gdb'),layer=f'T{tp}',columns=[],bbox=(sb[0]-.00002,sb[1]-.00002,sb[2]+.00002,sb[3]+.00002),return_fids=True)
                for targetfid,wkb in zip(tf,tg):
                    k=(np.uint64(tp)<<np.uint64(32))|np.uint64(targetfid)
                    at=np.searchsorted(sk,k);targetrow=int(order[at])
                    if targetrow==row:continue
                    t=shapely.make_valid(shapely.from_wkb(wkb))
                    if t.is_empty:continue
                    tm=shapely.set_precision(transform(tr.transform,t),1.0)
                    length=sm.boundary.intersection(tm.boundary).length
                    if length<=0:continue
                    count+=1
                    for yy,a in boundary.items():
                        coef=int(label[row]>0 and label[row]<yy)-int(oldlabel[row]>0 and oldlabel[row]<yy)
                        if coef:a[targetrow]+=coef*length
            if (it+1)%500==0:log(f'T{sourceprovince}: corrected boundaries for {it+1}/{len(rows)} changed sources')
        edgecounts.append(dict(source_partition=int(sourceprovince),changed_units=len(rows),shared_boundary_pairs=count))
        del sourcegeoms;gc.collect()
    checks=[]
    for yy,a in boundary.items():
        neg=int((a<-.01).sum());minimum=float(np.nanmin(a));checks.append(dict(year=yy,negative_after_correction=neg,min_value=minimum))
        # Keep problematic negative corrections as missing, never silently force them to zero.
        a[a<-.01]=np.nan;a[(a<0)&(a>=-.01)]=0;a.flush()
    outcsv(pd.DataFrame(checks),'11_边界长度修正数值诊断.csv');outcsv(pd.DataFrame(edgecounts),'12_边界修正逐分区记录.csv')
    js(W/'cache/history_features_complete.json',dict(pressure='complete',boundary='old geometry baseline plus changed-label corrections; 1 m projected grid tolerance; inconsistent negative entries set missing and audited',changed_units=len(changes),boundary_checks=checks))
    log('Historical exposure correction completed')
if __name__=='__main__':run()
