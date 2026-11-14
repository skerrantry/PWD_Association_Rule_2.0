from common import *
import pickle

def splitmix64(x):
    with np.errstate(over='ignore'):
      x=np.asarray(x,dtype=np.uint64)+np.uint64(0x9E3779B97F4A7C15)
      x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
      x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
      return x^(x>>np.uint64(31))

class Store:
    def __init__(self):
      marker=json.loads((C/'numeric_cache_manifest.json').read_text(encoding='utf-8'))
      assert marker['completed'] and marker['source']==source_metadata(RAW)
      self.pred=readcsv(A/'02_predicates/predicate_definitions_draft.csv')
      self.rows=self.pred.set_index('predicate_id').to_dict('index')
      self.by_source={s:df.index.to_list() for s,df in self.pred.set_index('predicate_id').groupby('source_field')}
      self.lookup={}
      for item,p in self.rows.items():
        token=re.sub(r'_(20\d{2})_', '_',item)
        year=int(p['year']) if pd.notna(p['year']) else 0
        self.lookup[(token,year)]=item
      self.data={f:np.load(C/f'{f}.npy',mmap_mode='r') for f in marker['fields']}
      self.shoufa=np.nan_to_num(self.data['SHOUFA_YEA'],nan=0).round().astype(np.int32)
      self.keys=np.load(C/'source_key.npy',mmap_mode='r')
    def item_for(self,token,year):
      return self.lookup.get((token,year),self.lookup.get((token,0)))
    def match(self,item,values=None,cuts=None):
      p=self.rows[item];field=p['source_field']
      v=np.asarray(self.data[field]) if values is None else values
      valid=np.isfinite(v)
      if field=='FO':
        if item.startswith('FO_detail_'):return valid&(v==int(item.rsplit('_',1)[1]))
        return valid&np.isin(v,[10,11,12,13] if item.endswith('natural') else [20,21,22,23,24])
      if field=='slope':
        v=np.where(np.isin(v,[1,2,3,4,5,6]),v*10,v)
        return valid&(v==int(item.rsplit('_',1)[1]))
      if field.startswith('GDP_'):valid&=v>=0
      family=p['family'];lo=p['lower'];hi=p['upper']
      if cuts is not None and family in cuts:
        m=re.search(r'_(posQ|Q)(\d)$',item)
        if m:
          positive=m[1]=='posQ';rank=int(m[2]);bound=cuts[family]
          lo=(0. if positive else float('nan')) if rank==1 else bound[rank-2]
          hi=float('nan') if rank==len(bound)+1 else bound[rank-1]
      if p['predicate_type']=='zero_inflated':return valid&(v==0)
      if pd.isna(lo) and pd.notna(hi):return valid&(v<=hi)
      if pd.notna(lo) and pd.isna(hi):return valid&(v>lo)
      if pd.notna(lo) and pd.notna(hi):
        if lo==0 and hi==0:return valid&(v==0)
        return valid&(v>lo)&(v<=hi)
      return np.zeros(len(v),dtype=bool)
    def cuts(self,train_years,scope):
      sample=splitmix64(self.keys^np.uint64(SEED))%np.uint64(10000)<100
      idx=np.flatnonzero(sample);result={};rows=[]
      for family in ['GDP','Bio8','Bio14','Bio18','inf_press_3km']:
        pooled=[]
        for y in train_years:
          v=np.asarray(self.data[f'{family}_{y}'][idx]);ok=np.isfinite(v)
          if family=='GDP':ok&=v>=0
          if family=='inf_press_3km':ok&=v>0
          pooled.append(v[ok])
        values=np.concatenate(pooled);qq=[.25,.5,.75] if family=='inf_press_3km' else [.2,.4,.6,.8]
        if len(values)==0:raise RuntimeError(f'No training values for {family} / {scope}')
        bounds=np.quantile(values,qq).tolist();result[family]=bounds
        for quantile,value in zip(qq,bounds):
          rows.append(dict(scope=scope,train_years=';'.join(map(str,train_years)),family=family,
            quantile=quantile,boundary=value,pooled_valid_values=len(values),source_units_sampled=len(idx),
            seed=SEED,method='SplitMix64 source-key 1% sample, outcome-blind training-year pooled quantiles'))
      return result,rows
    def score(self,year,families,cuts=None):
      family_tokens={f:tuple(f.split(';')) for f in families}
      tokens=sorted({t for f in family_tokens.values() for t in f})
      risk=(self.shoufa==0)|(self.shoufa>=year);positive=self.shoufa==year
      posidx=np.flatnonzero(positive);total=int(risk.sum());pos=int(positive.sum())
      bitmaps={};pbits={}
      for token in tokens:
        item=self.item_for(token,year)
        if item is None:bitmaps[token]=0;pbits[token]=0;continue
        mask=self.match(item,cuts=cuts)
        pbits[token]=packed_int(mask[posidx]);mask&=risk;bitmaps[token]=packed_int(mask)
      out=[]
      # Consecutive prefixes share intersections without retaining gigabytes of intermediate masks.
      pair=None;b0=p0=0;triple=None;b1=p1=0
      for family in sorted(families,key=lambda f:family_tokens[f]):
        ts=family_tokens[family]
        if ts[:2]!=pair:
          pair=ts[:2];b0=bitmaps[ts[0]]&bitmaps[ts[1]];p0=pbits[ts[0]]&pbits[ts[1]];triple=None
        b=b0;p=p0
        if len(ts)>=3:
          if ts[:3]!=triple:
            triple=ts[:3];b1=b0&bitmaps[ts[2]];p1=p0&pbits[ts[2]]
          b=b1;p=p1
        if len(ts)==4:b&=bitmaps[ts[3]];p&=pbits[ts[3]]
        out.append(dict(rule_family=family,year=year,**metrics(b.bit_count(),p.bit_count(),total,pos)))
      return pd.DataFrame(out)

def build_window_transactions(store,window):
    path=C/f'window_{window}_transaction_bits.pkl'
    if path.exists():
      with path.open('rb') as f:return pickle.load(f)
    train,validation=WINDOWS[window];t=time.time();sh=store.shoufa
    h=splitmix64(store.keys^np.uint64(SEED+window*10007))
    m=np.clip(np.where(sh==0,len(train),sh-2001),0,len(train))
    neg=np.flatnonzero((h%np.uint64(10000)<1000)&(m>0))
    h2=splitmix64(store.keys[neg]^np.uint64(SEED+window*10007+991))
    negyear=2001+(h2%m[neg].astype(np.uint64)).astype(np.int32)
    pos=np.flatnonzero(np.isin(sh,train))
    rows=np.concatenate([pos,neg]);year=np.concatenate([sh[pos],negyear]);nt=len(rows);np_=len(pos)
    assert np.isin(year,train).all() and np.all(negyear<np.where(sh[neg]==0,9999,sh[neg]))
    fullbits={};posbits={};ntypes=0
    for source,items in store.by_source.items():
      match=re.search(r'_(20\d{2})$',source)
      if match and int(match[1]) not in train:continue
      idx=np.flatnonzero(year==int(match[1])) if match else np.arange(nt)
      values=np.asarray(store.data[source][rows[idx]])
      for item in items:
        mask=np.zeros(nt,dtype=bool);mask[idx]=store.match(item,values)
        fullbits[item]=packed_int(mask);posbits[item]=packed_int(mask[:np_])
      ntypes+=1
      if ntypes%20==0:log('windows',f'window {window}: encoded {ntypes} source fields')
    obj=dict(total=nt,positive_count=np_,positive_bits=posbits,full_bits=fullbits,train_years=train,
      source=f'raw numeric cache; SplitMix64 source-key salts {SEED+window*10007}/{SEED+window*10007+991}',
      negative_source_units=len(neg),negative_selection_probability=.10,
      negative_record_probability='0.10 / number of eligible training-negative years',
      validation_years=validation)
    np.savez_compressed(C/f'window_{window}_transaction_indices.npz',source_row=rows,year=year,
                       positive_prefix_count=np.asarray([np_]))
    from miner import save_pickle
    save_pickle(path,obj)
    log('windows',f'window {window}: transactions={nt:,}, positives={np_:,}; build {time.time()-t:.1f}s')
    return obj
