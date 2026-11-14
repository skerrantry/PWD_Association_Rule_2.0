from env import *
from audit_upstream import get
from scipy.spatial import cKDTree
from pyproj import Transformer
import pyogrio,shapely,gc
def run():
    mapping=pd.read_csv(O/'核查与数据表/04_源FID到分析小班修复映射.csv')
    source=pd.read_pickle(W/'cache/infected_attributes.pkl').set_index('physical_fid')
    keys=np.load(OLD/'source_key.npy',mmap_mode='r');order=np.load(OLD/'source_key_order.npy',mmap_mode='r');sk=keys[order]
    x=np.load(W/'cache/BAN_X.npy',mmap_mode='r');y=np.load(W/'cache/BAN_Y.npy',mmap_mode='r')
    label=np.load(W/'cache/repaired_first_year.npy').copy();extra=[];counts=[]
    for pc in [35,44,51,61]:
        un=mapping[(mapping.province==pc)&(mapping.target_source_row<0)].copy()
        loc=np.flatnonzero((keys>>np.uint64(32))==pc)
        tr=Transformer.from_crs(4326,3857,always_xy=True)
        xx,yy=tr.transform(x[loc],y[loc]);tree=cKDTree(np.column_stack([xx,yy]))
        log(f'T{pc}: checking polygon overlap for {len(un):,} unresolved records')
        for at in range(0,len(un),200):
            z=un.iloc[at:at+200];m,f,g,a=pyogrio.raw.read(str(F/'80_Infectedxiaoban_2020/merged.shp'),columns=[],fids=z.source_fid.astype(int).tolist(),return_fids=True)
            sg={int(fid):shapely.make_valid(shapely.from_wkb(gg)) for fid,gg in zip(f,g)}
            px,py=tr.transform(z.source_lon.to_numpy(),z.source_lat.to_numpy());dist,jj=tree.query(np.column_stack([px,py]),k=64,distance_upper_bound=5000,workers=4)
            good=np.isfinite(dist);wanted=np.unique(loc[jj[good]])
            oid=(keys[wanted]&np.uint64(0xffffffff)).astype(np.int64)
            mm,ff,gg,aa=pyogrio.raw.read(str(F/'Sub_Pine_All.gdb'),layer=f'T{pc}',columns=[],fids=oid.tolist(),return_fids=True)
            targets={int(fid):shapely.make_valid(shapely.from_wkb(v)) for fid,v in zip(ff,gg)}
            for k,(_,r) in enumerate(z.iterrows()):
                s=sg[int(r.source_fid)];accepted=[]
                if s.is_empty or s.area<=0:continue
                for j in jj[k][good[k]]:
                    row=int(loc[j]);o=int(keys[row]&np.uint64(0xffffffff));t=targets[o]
                    if not s.intersects(t) or t.area<=0:continue
                    area=s.intersection(t).area;fs=area/s.area;ft=area/t.area
                    if (fs>=.5 and ft>=.1) or (ft>=.5 and fs>=.1):accepted.append((row,o,fs,ft))
                for row,o,fs,ft in accepted:
                    year=int(r.source_year);label[row]=min(int(label[row]),year) if label[row]>0 else year
                    extra.append(dict(province=pc,source_fid=int(r.source_fid),source_year=year,target_source_row=row,target_oid=o,source_overlap_fraction=fs,target_overlap_fraction=ft,method='polygon_overlap_50percent_with_10percent_reciprocal_minimum'))
            if (at//200)%5==0:log(f'T{pc}: unresolved polygon batch {min(at+200,len(un))}/{len(un)}; cumulative links {len(extra)}')
        resolved={r['source_fid'] for r in extra if r['province']==pc}
        counts.append(dict(province=pc,unresolved_before=len(un),resolved_by_polygon_overlap=len(resolved),still_unmatched=len(un)-len(resolved)))
        outcsv(pd.DataFrame(extra),'06_第二轮多边形重叠匹配明细.csv');outcsv(pd.DataFrame(counts),'07_第二轮空间匹配摘要.csv')
        np.save(W/'cache/spatially_repaired_first_year.npy',label)
        del tree,targets;gc.collect()
    js(W/'cache/spatial_refinement_complete.json',dict(summary=counts,positive_units=int(((label>=2001)&(label<=2020)).sum()),overlap_rule='at least 50% of either polygon and at least 10% of the other; spatial harmonization, not evidence of historical identity'))
if __name__=='__main__':run()
