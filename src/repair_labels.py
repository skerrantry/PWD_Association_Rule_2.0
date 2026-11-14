from env import *
from audit_upstream import get
from scipy.spatial import cKDTree
import pyogrio,shapely,gc
from pyproj import Transformer
from collections import defaultdict,Counter
KEY=['SHENG','XIAN','XIANG','CUN','LIN_YE_JU','LIN_CHANG','LIN_BAN','XIAO_BAN']
def norm(x):
    if pd.isna(x):return ''
    s=str(x).strip()
    if s.endswith('.0'):s=s[:-2]
    return s.lstrip('0') or ('0' if s else '')
def keyrows(d,upper=False):
    cols=KEY if upper else [x.lower() for x in KEY]
    arr=[d[x].map(norm).to_numpy() for x in cols]
    # Match county, last three township/village digits, forest compartment and subcompartment.
    for j in [2,3]:arr[j]=np.array([norm(x[-3:]) if x.isdigit() else x for x in arr[j]],object)
    return list(zip(*(arr[i] for i in [0,1,2,3,6,7])))
def run():
    source=pd.read_pickle(W/'cache/infected_attributes.pkl')
    keys=np.load(OLD/'source_key.npy',mmap_mode='r');order=np.load(OLD/'source_key_order.npy',mmap_mode='r');sk=keys[order]
    old=np.nan_to_num(np.load(OLD/'SHOUFA_YEA.npy',mmap_mode='r'),nan=0).astype(np.int16)
    new=old.copy();x=np.load(W/'cache/BAN_X.npy',mmap_mode='r+');y=np.load(W/'cache/BAN_Y.npy',mmap_mode='r+')
    audit=pd.read_csv(O/'核查与数据表/01_病例源编号碰撞逐省审计.csv')
    affected=audit[(audit.duplicate_OBJECTID_rows>0)&(audit.positive_2001_2020>0)].province.astype(int).tolist()
    mapping=[];summ=[]
    for pc in affected:
        log(f'Repair T{pc}: reading original administrative keys and existing source linkage')
        d,fids,_,_=get(F/'Sub_Pine_All.gdb',layer=f'T{pc}',cols=[k.lower() for k in KEY]+['infect_srcoid','infect_method','SHOUFA_YEA','BAN_X','BAN_Y','center_x','center_y'])
        loc=order[np.searchsorted(sk,(np.uint64(pc)<<np.uint64(32))|fids.astype(np.uint64))]
        for col,arr in [('BAN_X',x),('BAN_Y',y)]:
            v=pd.to_numeric(d[col],errors='coerce').to_numpy()
            fallback=pd.to_numeric(d['center_x' if col=='BAN_X' else 'center_y'],errors='coerce').to_numpy()
            v=np.where(np.isfinite(v)&(v!=0),v,fallback);arr[loc]=v
        sx=np.asarray(x[loc]);sy=np.asarray(y[loc]);valid=np.isfinite(sx)&np.isfinite(sy)&(sx>70)&(sx<140)&(sy>15)&(sy<55)
        tr=Transformer.from_crs(4326,3857,always_xy=True)
        xx,yy=tr.transform(sx[valid],sy[valid]);vrows=np.flatnonzero(valid);tree=cKDTree(np.column_stack([xx,yy]))
        s=source[(pd.to_numeric(source.SHENG,errors='coerce')==pc)&source.SHOUFA_YEA.between(2001,2020)].copy()
        src_counts=source[pd.to_numeric(source.SHENG,errors='coerce')==pc].OBJECTID.value_counts()
        targets=defaultdict(list)
        for j,k in enumerate(keyrows(d)):targets[k].append(j)
        oldlinks=defaultdict(list)
        for j,(oid,yr) in enumerate(zip(d.infect_srcoid,d.SHOUFA_YEA)):
            if pd.notna(oid) and pd.notna(yr) and yr>0:oldlinks[int(oid)].append(j)
        scx=pd.to_numeric(s.center_x,errors='coerce').to_numpy();scy=pd.to_numeric(s.center_y,errors='coerce').to_numpy()
        scx=np.where(np.isfinite(scx),scx,pd.to_numeric(s.O_LON_P,errors='coerce'));scy=np.where(np.isfinite(scy),scy,pd.to_numeric(s.O_LAT_P,errors='coerce'))
        px,py=tr.transform(scx,scy)
        distances,near=tree.query(np.column_stack([px,py]),k=8,distance_upper_bound=1500,workers=4)
        candidates={};assign={};method={}
        for k,(idx,row) in enumerate(s.iterrows()):
            oid=int(row.OBJECTID)
            if src_counts.get(oid,0)==1 and len(oldlinks.get(oid,[]))==1:
                assign[k]=oldlinks[oid][0];method[k]='retained_unique_source_link';continue
            cand=targets.get(keyrows(s.iloc[[k]],True)[0],[])
            good=[j for j in cand if valid[j]]
            if good:
                tx,ty=tr.transform(sx[good],sy[good]);dd=np.hypot(tx-px[k],ty-py[k]);best=int(np.argmin(dd))
                if dd[best]<=1000 and (len(dd)==1 or np.sort(dd)[1]>dd[best]+20):
                    assign[k]=good[best];method[k]='normalized_admin_and_coordinate';continue
            cand2=[int(vrows[j]) for dist,j in zip(distances[k],near[k]) if np.isfinite(dist) and j<len(vrows)]
            candidates[k]=list(dict.fromkeys(cand+ cand2))
        # Point-in-polygon confirmation for unmatched administrative records.
        wanted=sorted({j for cand in candidates.values() for j in cand})
        geometries={}
        for at in range(0,len(wanted),2000):
            jj=wanted[at:at+2000]
            mm,fi,gg,aa=pyogrio.raw.read(str(F/'Sub_Pine_All.gdb'),layer=f'T{pc}',columns=[],fids=fids[jj],return_fids=True)
            for fid,g in zip(fi,gg):geometries[int(fid)]=shapely.from_wkb(g)
        for k,cand in candidates.items():
            point=shapely.Point(scx[k],scy[k]);inside=[]
            for j in cand:
                g=geometries.get(int(fids[j]))
                if g is not None and not g.is_empty and g.covers(point):inside.append(j)
            if len(inside)==1:assign[k]=inside[0];method[k]='source_centroid_in_unique_target_polygon'
            elif len(inside)>1:
                # Identical overlapping base/appended copies are ambiguous; do not select by outcome.
                method[k]='ambiguous_multiple_containing_polygons'
            else:method[k]='unmatched_no_confirmed_polygon'
        new[loc]=0
        for k,(idx,row) in enumerate(s.iterrows()):
            j=assign.get(k);globalrow=int(loc[j]) if j is not None else -1
            mapping.append(dict(province=pc,source_fid=int(row.physical_fid),attribute_OBJECTID=int(row.OBJECTID),source_year=int(row.SHOUFA_YEA),target_source_row=globalrow,target_oid=int(fids[j]) if j is not None else '',method=method.get(k,'unmatched'),source_lon=scx[k],source_lat=scy[k]))
            if j is not None:
                v=int(row.SHOUFA_YEA);new[globalrow]=min(int(new[globalrow]),v) if new[globalrow]>0 else v
        mapped=[r for r in mapping if r['province']==pc]
        # Failed source linkage means unknown coverage, not verified absence: quarantine relevant counties.
        unmatched=s.iloc[[k for k in range(len(s)) if k not in assign]]
        badcounties={norm(v) for v in unmatched.XIAN}
        quarantine=np.array([norm(v) in badcounties for v in d.xian])
        np.save(W/'cache'/f'T{pc}_unmatched_county_rows.npy',loc[quarantine])
        summ.append(dict(province=pc,source_events=len(s),matched_source_records=len(assign),unmatched_source_records=len(s)-len(assign),old_positive_units=int(((old[loc]>=2001)&(old[loc]<=2020)).sum()),new_positive_units=int((new[loc]>0).sum()),quarantined_counties=len(badcounties),quarantined_units=int(quarantine.sum())))
        outcsv(pd.DataFrame(mapping),'04_源FID到分析小班修复映射.csv');outcsv(pd.DataFrame(summ),'05_标签修复逐省摘要.csv')
        np.save(W/'cache'/'repaired_first_year.npy',new)
        log(f'T{pc}: {summ[-1]}')
        del d,tree,geometries;gc.collect()
    x.flush();y.flush()
    js(W/'cache'/'label_repair_complete.json',dict(affected=affected,old_positive=int(((old>=2001)&(old<=2020)).sum()),new_positive=int(((new>=2001)&(new<=2020)).sum()),summary=summ))
if __name__=='__main__':run()
