from env import *
import pyogrio,shapely
from pyproj import Transformer
def run():
    keys=np.load(OLD/'source_key.npy',mmap_mode='r');pc=(keys>>np.uint64(32)).astype(np.uint8)
    x=np.load(W/'cache/BAN_X.npy',mmap_mode='r+');y=np.load(W/'cache/BAN_Y.npy',mmap_mode='r+')
    invalid=~(np.isfinite(x)&np.isfinite(y)&(x>=70)&(x<=140)&(y>=15)&(y<=55))
    label=np.load(W/'cache/analysis_first_year.npy');rows=np.flatnonzero(invalid|(label>0));records=[]
    previous=json.loads((W/'cache/reference_coordinates_fixed.json').read_text(encoding='utf-8')) if (W/'cache/reference_coordinates_fixed.json').exists() else {}
    original_invalid_count=previous.get('original_invalid_coordinates_restored',previous.get('restored_units',int(invalid.sum())))
    log(f'Reference coordinates: recomputing {len(rows):,} event or invalid points from their own geometry')
    for province in np.unique(pc[rows]):
        rr=rows[pc[rows]==province];lookup={int(keys[row]&np.uint64(0xffffffff)):int(row) for row in rr}
        for at in range(0,len(rr),2000):
            sub=rr[at:at+2000];oid=(keys[sub]&np.uint64(0xffffffff)).astype(np.int64)
            m,fi,gg,a=pyogrio.raw.read(str(F/'Sub_Pine_All.gdb'),layer=f'T{province}',columns=[],fids=oid.tolist(),return_fids=True)
            geoms=shapely.make_valid(shapely.from_wkb(gg));centers=shapely.centroid(geoms)
            cx=shapely.get_x(centers);cy=shapely.get_y(centers)
            for fid,xx,yy in zip(fi,cx,cy):
                row=lookup[int(fid)];records.append(dict(source_row=row,partition=int(province),first_year=int(label[row]),old_x=float(x[row]),old_y=float(y[row]),restored_x=float(xx),restored_y=float(yy),method='centroid_of_own_archived_WGS84_polygon'))
                x[row]=xx;y[row]=yy
        log(f'T{province}: restored {len(rr):,} reference coordinates')
    assert np.all(np.isfinite(x)&np.isfinite(y)&(x>=70)&(x<=140)&(y>=15)&(y<=55))
    x.flush();y.flush();outcsv(pd.DataFrame(records),'34_零值或异常参考坐标自身几何修复明细.csv')
    js(W/'cache/reference_coordinates_fixed.json',dict(original_invalid_coordinates_restored=original_invalid_count,recomputed_event_or_invalid_points=len(rows),recomputed_recorded_event_units=int(((label[rows]>=2001)&(label[rows]<=2020)).sum()),all_points_in_geographic_range=True,method='recompute every recorded-event point and every invalid point from its own archived WGS84 polygon centroid; retain valid non-event inventory centroids; no neighbour-location imputation',area_unit='stored management area is hectares, not square metres; sampled geometric ratio median approximately 0.0001'))
if __name__=='__main__':run()
