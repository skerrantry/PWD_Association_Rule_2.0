"""Recompute all history contributions for targets whose boundaries may be affected."""
from env import *
import sqlite3,pyogrio,shapely,gc,warnings
from pyproj import Transformer
from shapely.ops import transform
warnings.filterwarnings('ignore',category=DeprecationWarning,module='__main__')
def run():
    assert (W/'cache/final_linkage_qa_complete.json').exists()
    db=sqlite3.connect(W/'cache/geometry_bbox.sqlite')
    keys=np.load(OLD/'source_key.npy',mmap_mode='r');order=np.load(OLD/'source_key_order.npy',mmap_mode='r');sk=keys[order]
    pc=(keys>>np.uint64(32)).astype(np.uint8);n=len(keys)
    old=np.nan_to_num(np.load(OLD/'SHOUFA_YEA.npy',mmap_mode='r'),nan=0).astype(int);new=np.load(W/'cache/analysis_first_year.npy')
    changed=np.flatnonzero(old!=new)
    affected=set()
    for row in changed:
        b=db.execute('SELECT minx,maxx,miny,maxy FROM bbox WHERE source_row=?',(int(row),)).fetchone()
        if b:
            affected.update(int(z[0]) for z in db.execute('SELECT source_row FROM bbox WHERE minx<=? AND maxx>=? AND miny<=? AND maxy>=?',(b[1]+.00002,b[0]-.00002,b[3]+.00002,b[2]-.00002)))
    affected=np.array(sorted(affected),dtype=np.int64);np.save(W/'cache/boundary_recomputed_target_rows.npy',affected)
    arrays={y:np.lib.format.open_memmap(W/'cache'/f'inf_by_{y}.npy',mode='w+',dtype=np.float64,shape=(n,)) for y in range(2001,2021)}
    for y,a in arrays.items():a[:]=np.load(OLD/f'inf_by_{y}.npy',mmap_mode='r');a[affected]=0
    log(f'Recomputing complete shared-boundary history for {len(affected):,} affected targets')
    summaries=[]
    for province in np.unique(pc[affected]):
        targets=affected[pc[affected]==province];info=json.loads((W/'cache'/f'T{province}_info.json').read_text(encoding='utf-8'));b=info['total_bounds']
        tr=Transformer.from_crs(4326,f'+proj=aeqd +lat_0={(b[1]+b[3])/2} +lon_0={(b[0]+b[2])/2} +datum=WGS84 +units=m',always_xy=True)
        cache={};edges=0
        def fetch(rr):
            missing=np.array([int(r) for r in rr if int(r) not in cache],dtype=np.int64)
            for tp in np.unique(pc[missing]):
                ii=missing[pc[missing]==tp];ids=(keys[ii]&np.uint64(0xffffffff)).astype(np.int64)
                mm,ff,gg,aa=pyogrio.raw.read(str(F/'Sub_Pine_All.gdb'),layer=f'T{tp}',columns=[],fids=ids.tolist(),return_fids=True)
                for fid,wkb in zip(ff,gg):
                    k=(np.uint64(tp)<<np.uint64(32))|np.uint64(fid);row=int(order[np.searchsorted(sk,k)])
                    g=shapely.make_valid(shapely.from_wkb(wkb));cache[row]=shapely.set_precision(transform(tr.transform,g),1.0)
        for at in range(0,len(targets),500):
            chunk=targets[at:at+500];fetch(chunk)
            for row in chunk:
                bb=db.execute('SELECT minx,maxx,miny,maxy FROM bbox WHERE source_row=?',(int(row),)).fetchone()
                cand=np.array([z[0] for z in db.execute('SELECT source_row FROM bbox WHERE minx<=? AND maxx>=? AND miny<=? AND maxy>=?',(bb[1]+.00002,bb[0]-.00002,bb[3]+.00002,bb[2]-.00002))],dtype=np.int64)
                cand=cand[(new[cand]>0)&(new[cand]<2020)&(cand!=row)]
                fetch(cand);t=cache[int(row)]
                if t.is_empty:continue
                for src in cand:
                    s=cache[int(src)]
                    if s.is_empty:continue
                    length=t.boundary.intersection(s.boundary).length
                    if length<=0:continue
                    edges+=1
                    for y in range(max(2001,int(new[src])+1),2021):arrays[y][row]+=length
            if len(cache)>75000:cache.clear()
            if (at//500)%4==0:log(f'Boundary full T{province}: {min(at+500,len(targets)):,}/{len(targets):,}')
        summaries.append(dict(partition=int(province),recomputed_targets=len(targets),positive_shared_boundary_pairs=edges))
        for a in arrays.values():a.flush()
        outcsv(pd.DataFrame(summaries),'17_受影响目标小班完整边界历史重算.csv')
        del cache;gc.collect()
    checks=[]
    for y,a in arrays.items():
        assert np.isfinite(a).all() and (a>=0).all()
        checks.append(dict(year=y,recomputed_target_records=len(affected),negative_values=0,missing_values=0,total_boundary_length=float(a.sum())))
    outcsv(pd.DataFrame(checks),'18_边界完整重算验收.csv')
    js(W/'cache/boundary_full_complete.json',dict(affected_targets=len(affected),changed_first_year_units=len(changed),nonnegative=True,finite=True,method='complete neighbor-history recomputation for targets intersecting changed-source envelopes, local AEQD and 1m coordinate grid; unchanged targets retain original baseline',summary=summaries))
    log('Complete affected-target boundary reconstruction passed finite/nonnegative checks')
if __name__=='__main__':run()
