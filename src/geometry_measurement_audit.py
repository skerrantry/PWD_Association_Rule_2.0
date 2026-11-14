from env import *
import pyogrio,shapely
from pyproj import Transformer
from shapely.ops import transform
def run():
    keys=np.load(OLD/'source_key.npy',mmap_mode='r');pc=(keys>>np.uint64(32)).astype(np.uint8)
    area=np.load(OLD/'Area.npy',mmap_mode='r');x=np.load(W/'cache/BAN_X.npy',mmap_mode='r');y=np.load(W/'cache/BAN_Y.npy',mmap_mode='r')
    label=np.load(W/'cache/analysis_first_year.npy');rows=[];rng=np.random.default_rng(20261111)
    for province in np.unique(pc):
        selected=[]
        for state in [False,True]:
            ids=np.flatnonzero((pc==province)&((label>=2001)&(label<=2020) if state else label==0))
            if len(ids):selected+=rng.choice(ids,size=min(30,len(ids)),replace=False).tolist()
        if not selected:continue
        selected=np.array(selected,dtype=np.int64);oids=(keys[selected]&np.uint64(0xffffffff)).astype(np.int64)
        m,fi,gg,a=pyogrio.raw.read(str(F/'Sub_Pine_All.gdb'),layer=f'T{province}',columns=[],fids=oids.tolist(),return_fids=True)
        where={int(o):int(r) for o,r in zip(oids,selected)}
        info=json.loads((W/'cache'/f'T{province}_info.json').read_text(encoding='utf-8'));b=info['total_bounds']
        tr=Transformer.from_crs(4326,f'+proj=aeqd +lat_0={(b[1]+b[3])/2} +lon_0={(b[0]+b[2])/2} +datum=WGS84 +units=m',always_xy=True)
        for oid,wkb in zip(fi,gg):
            row=where[int(oid)];g=shapely.make_valid(shapely.from_wkb(wkb));gm=transform(tr.transform,g);px,py=tr.transform(x[row],y[row]);cen=gm.centroid
            rows.append(dict(partition=int(province),source_row=row,first_year=int(label[row]),stored_Area=float(area[row]),geometry_area_m2=float(gm.area),stored_to_geometry_area_ratio=float(area[row]/gm.area) if gm.area else np.nan,reference_to_polygon_centroid_distance_m=float(np.hypot(cen.x-px,cen.y-py))))
    df=pd.DataFrame(rows);outcsv(df,'33_面积单位及压力参考点抽样审计.csv')
    summary=dict(sampled_records=len(df),area_ratio_quantiles=df.stored_to_geometry_area_ratio.quantile([0,.05,.5,.95,1]).to_dict(),point_distance_quantiles_m=df.reference_to_polygon_centroid_distance_m.quantile([0,.5,.95,.99,1]).to_dict(),area_ratio_below_point01=int((df.stored_to_geometry_area_ratio<.01).sum()),area_ratio_above100=int((df.stored_to_geometry_area_ratio>100).sum()))
    js(W/'cache/geometry_measurement_audit.json',summary);log(f'Geometry measurement audit: {summary}')
if __name__=='__main__':run()
