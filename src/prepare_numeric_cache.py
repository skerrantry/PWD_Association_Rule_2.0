from common import *

def run():
    marker=C/'numeric_cache_manifest.json'
    if marker.exists():
        m=json.loads(marker.read_text(encoding='utf-8'))
        if m.get('completed') and m['source']==source_metadata(RAW):
            log('cache','Verified completed numeric cache; reuse');return
    pred=readcsv(A/'02_predicates/predicate_definitions_draft.csv')
    fields=sorted(set(pred.source_field)|{'SHOUFA_YEA'})
    arrays={f:np.lib.format.open_memmap(C/f'{f}.npy',mode='w+',dtype='float64',shape=(N,)) for f in fields}
    keys=np.lib.format.open_memmap(C/'source_key.npy',mode='w+',dtype='uint64',shape=(N,))
    at=0;started=time.time();counts={};sample=[]
    for chunkno,chunk in enumerate(pd.read_csv(RAW,usecols=fields+['province_fc','OBJECTID'],chunksize=100_000,low_memory=False),1):
        end=at+len(chunk)
        for field in fields:arrays[field][at:end]=pd.to_numeric(chunk[field],errors='coerce').to_numpy(dtype=np.float64,na_value=np.nan)
        fc=chunk.province_fc.astype(str).str.removeprefix('T').astype(np.uint64).to_numpy()
        oid=pd.to_numeric(chunk.OBJECTID,errors='raise').to_numpy(dtype=np.uint64)
        keys[at:end]=(fc<<np.uint64(32))|oid
        yy=pd.to_numeric(chunk.SHOUFA_YEA,errors='coerce').fillna(0).round().astype(int)
        for year,n in yy.value_counts().items():counts[str(year)]=counts.get(str(year),0)+int(n)
        if len(sample)<10:sample+=chunk.head(1).fillna('').to_dict('records')
        at=end
        if chunkno%10==0 or at==N:
            log('cache',f'Numeric cache rows {at:,}/{N:,}; elapsed {(time.time()-started)/60:.1f} min')
            save_json(C/'numeric_cache_progress.json',dict(rows=at,total=N,elapsed_seconds=time.time()-started))
    assert at==N and sum(counts.get(str(y),0) for y in YEARS)==237840
    for arr in arrays.values():arr.flush()
    keys.flush()
    order=np.argsort(keys)
    assert np.all(np.diff(np.asarray(keys)[order])>0)
    np.save(C/'source_key_order.npy',order)
    save_json(marker,dict(completed=True,source=source_metadata(RAW),rows=at,fields=fields,
      first_recorded_year_counts=counts,sample_rows=sample,elapsed_seconds=time.time()-started))
    log('cache',f'Numeric cache complete: {len(fields)} fields; {at:,} rows')

if __name__=='__main__':run()
