from env import *
import pyogrio,shapely,sqlite3
def run():
    path=W/'cache/geometry_bbox.sqlite';db=sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS bbox USING rtree(source_row,minx,maxx,miny,maxy)')
    db.execute('CREATE TABLE IF NOT EXISTS completed(partition INTEGER PRIMARY KEY,rows INTEGER)')
    keys=np.load(OLD/'source_key.npy',mmap_mode='r');order=np.load(OLD/'source_key_order.npy',mmap_mode='r');sk=keys[order]
    total=0
    for infofile in (W/'cache').glob('T*_info.json'):
        code=infofile.stem[1:-5]
        if not code.isdigit():continue
        pc=int(code);info=json.loads(infofile.read_text(encoding='utf-8'))
        if db.execute('SELECT 1 FROM completed WHERE partition=?',(pc,)).fetchone():continue
        n=int(info['features']);log(f'Building independent polygon bounds index T{pc}: {n:,}')
        for at in range(0,n,10000):
            m,fi,gg,a=pyogrio.raw.read(str(F/'Sub_Pine_All.gdb'),layer=f'T{pc}',columns=[],skip_features=at,max_features=10000,return_fids=True)
            geoms=shapely.from_wkb(gg,on_invalid='warn');bounds=shapely.bounds(geoms)
            loc=order[np.searchsorted(sk,(np.uint64(pc)<<np.uint64(32))|fi.astype(np.uint64))]
            records=[(int(row),float(b[0]),float(b[2]),float(b[1]),float(b[3])) for row,b in zip(loc,bounds) if np.isfinite(b).all()]
            db.executemany('INSERT OR REPLACE INTO bbox VALUES(?,?,?,?,?)',records)
        db.execute('INSERT OR REPLACE INTO completed VALUES(?,?)',(pc,n));db.commit();total+=n
        log(f'T{pc} index complete')
    js(W/'cache/spatial_index_complete.json',dict(index=str(path),partitions=db.execute('SELECT count(*) FROM completed').fetchone()[0],indexed_polygons=db.execute('SELECT count(*) FROM bbox').fetchone()[0]))
    db.close();log('Independent spatial index complete')
if __name__=='__main__':run()
