from env import *
import pyogrio
from pyogrio.raw import read
def get(path,layer=None,cols=None,geom=False,**kw):
    m,f,g,a=read(str(path),layer=layer,columns=cols,read_geometry=geom,return_fids=True,**kw)
    return pd.DataFrame(dict(zip(m['fields'],a))),f,g,m
def run():
    source=F/'80_Infectedxiaoban_2020'/'merged.shp'
    inf=pyogrio.read_info(source)
    js(W/'cache'/'source_info.json',{k:v.tolist() if hasattr(v,'tolist') else v for k,v in inf.items()})
    cols=[x for x in inf['fields'] if x.lower() in ['sheng','xian','xiang','cun','lin_ye_ju','lin_chang','lin_ban','xiao_ban','shoufa_yea','objectid','uniqueid','center_x','center_y','o_lon_p','o_lat_p','lon_p','lat_p','you_shi_sz']]
    df,fids,g,m=get(source,cols=cols)
    df.insert(0,'physical_fid',fids)
    df.to_pickle(W/'cache'/'infected_attributes.pkl')
    rows=[]
    for p,d in df.groupby('SHENG',dropna=False):
        rows.append(dict(province=p,records=len(d),distinct_attribute_OBJECTID=d.OBJECTID.nunique(dropna=False),distinct_physical_FID=d.physical_fid.nunique(),duplicate_OBJECTID_rows=int(d.OBJECTID.duplicated(keep=False).sum()),positive_2001_2020=int(d.SHOUFA_YEA.between(2001,2020).sum()),unique_uid=d.UniqueID.nunique()))
    outcsv(pd.DataFrame(rows),'01_病例源编号碰撞逐省审计.csv')
    log(f'Source audit complete: {len(df):,} records')
    keys=np.load(OLD/'source_key.npy',mmap_mode='r');order=np.load(OLD/'source_key_order.npy',mmap_mode='r')
    sortedkeys=keys[order];N=len(keys)
    fields=['BAN_X','BAN_Y','you_shi_sz','SHOUFA_YEA']
    arrays={x:np.lib.format.open_memmap(W/'cache'/f'{x}.npy',mode='w+',dtype=np.float64,shape=(N,)) for x in fields}
    for a in arrays.values():a[:]=np.nan
    provinces=(keys>>np.uint64(32)).astype(np.uint8)
    np.save(W/'cache'/'province.npy',provinces)
    joined=0;pr=[]
    layers=pyogrio.list_layers(F/'Sub_Pine_All.gdb')
    js(W/'cache'/'final_layers.json',layers.tolist())
    for layer,typ in layers:
        if not (str(layer).startswith('T') and str(layer)[1:].isdigit()):continue
        code=int(str(layer)[1:]);info=pyogrio.read_info(F/'Sub_Pine_All.gdb',layer=layer)
        js(W/'cache'/f'{layer}_info.json',{k:v.tolist() if hasattr(v,'tolist') else v for k,v in info.items()})
        d,ids,g,m=get(F/'Sub_Pine_All.gdb',layer=layer,cols=[x for x in fields if x in info['fields']])
        k=(np.uint64(code)<<np.uint64(32))|ids.astype(np.uint64)
        at=np.searchsorted(sortedkeys,k);assert np.array_equal(sortedkeys[at],k)
        loc=order[at]
        for field in fields:
            if field in d:arrays[field][loc]=pd.to_numeric(d[field],errors='coerce').to_numpy()
        vals=pd.to_numeric(d.SHOUFA_YEA,errors='coerce').fillna(0)
        for y,n in vals.value_counts().items():pr.append(dict(province=code,first_recorded_year=int(y),rows=int(n)))
        joined+=len(d);log(f'{layer}: {len(d):,} joined; {joined:,}/{N:,}')
    assert joined==N
    for a in arrays.values():a.flush()
    outcsv(pd.DataFrame(pr),'02_原最终库逐省首发年份.csv')
    sp=pd.DataFrame({'province':provinces,'code':arrays['you_shi_sz']}).groupby(['province','code'],dropna=False).size().reset_index(name='n')
    outcsv(sp,'03_原小班优势树种代码逐省.csv')
    js(W/'cache'/'audit_complete.json',dict(n=N,source_records=len(df),coordinates_valid=int((np.isfinite(arrays['BAN_X'])&np.isfinite(arrays['BAN_Y'])).sum())))
if __name__=='__main__':run()
