"""Rebuilt risk records, known-probability sampling and independent rediscovery runs."""
from env import *
import argparse,pickle,re,hashlib,math,gc
TASK_W=W;TASK_O=O
from predicates import Store as LegacyStore,splitmix64
import miner as legacy_miner
from common import packed_int,metrics,family_key
legacy_miner.W=TASK_W;legacy_miner.log=lambda step,msg:log(msg)
YEARS=list(range(2001,2021));SEED=20261111
FOLDS={i+1:list(range(2001+i,2021,5)) for i in range(5)}
def actual_expression(p):
    item=p['predicate_id'];field=p['source_field'];valid=f'isfinite({field})'
    if field=='FO':
        if item.startswith('FO_detail_'):return valid+f' and FO == {int(item.rsplit("_",1)[1])}'
        return valid+(' and FO in {10,11,12,13}' if item.endswith('natural') else ' and FO in {20,21,22,23,24}')
    if field=='slope':return f'isfinite(slope) and normalize_1_to_6_as_10_to_60(slope) == {int(item.rsplit("_",1)[1])}'
    if field.startswith('GDP_'):valid+=f' and {field} >= 0'
    lo=p['lower'];hi=p['upper']
    if p['predicate_type']=='zero_inflated' or (pd.notna(lo) and pd.notna(hi) and lo==0 and hi==0):cond=f'{field} == 0'
    elif pd.isna(lo) and pd.notna(hi):cond=f'{field} <= {hi!r}'
    elif pd.notna(lo) and pd.isna(hi):cond=f'{field} > {lo!r}'
    elif pd.notna(lo) and pd.notna(hi):cond=f'{lo!r} < {field} <= {hi!r}'
    else:cond='False'
    return valid+' and '+cond
class RebuiltStore(LegacyStore):
    def __init__(self):
        super().__init__()
        self.shoufa=np.load(TASK_W/'cache/analysis_first_year.npy')
        self.censor=np.load(TASK_W/'cache/linkage_censor_year.npy')
        self.county=np.load(OLD/'county_id.npy',mmap_mode='r')
        self.groups=pd.read_csv(OLD/'county_groups.csv',encoding='utf-8-sig',dtype={'province_group':str})
        lookup=self.groups.set_index('county_group_id').province_group.to_dict()
        names=sorted(set(lookup.values()));self.province_names=names
        codes={p:i for i,p in enumerate(names)}
        percounty=np.array([codes[lookup[i]] for i in range(len(lookup))],dtype=np.int16)
        self.province=percounty[self.county]
        observed=(self.shoufa>=2001)&(self.shoufa<=2020)&(self.shoufa<self.censor)
        self.active_provinces=np.flatnonzero(np.bincount(self.province[observed],minlength=len(names))>0).tolist()
        for year in YEARS:
            for prefix in ['inf_by','inf_press_3km']:
                self.data[f'{prefix}_{year}']=np.load(TASK_W/'cache'/f'{prefix}_{year}.npy',mmap_mode='r')
        # Original static/climate/GDP backgrounds stay unchanged because the author-defined cohort is unchanged.
        # Recalculate only annual positive-pressure quantiles after changing the historical labels.
        self.pred=self.pred.copy()
        for year in YEARS:
            field=f'inf_press_3km_{year}';v=np.asarray(self.data[field]);values=v[np.isfinite(v)&(v>0)]
            cuts=np.quantile(values,[.25,.5,.75]) if len(values) else np.array([np.nan]*3)
            idx=self.pred.source_field.eq(field)
            for j in self.pred.index[idx]:
                item=self.pred.at[j,'predicate_id'];match=re.search(r'posQ(\d)$',item)
                if match:
                    rank=int(match[1]);self.pred.at[j,'lower']=0 if rank==1 else cuts[rank-2]
                    self.pred.at[j,'upper']=np.nan if rank==4 else cuts[rank-1]
        self.pred['expression']=self.pred.apply(actual_expression,axis=1)
        category=self.pred.source_field.isin(['FO','slope'])
        equality=self.pred.predicate_type.eq('zero_inflated')|((self.pred.lower==0)&(self.pred.upper==0))
        self.pred['include_lower']=np.where(category,'not_applicable',np.where(equality,'True','False'))
        self.pred['include_upper']=np.where(category,'not_applicable',np.where(self.pred.upper.notna()|equality,'True','False'))
        self.pred['runtime_boundary_note']='numerical lower open / upper closed; equality predicates match exact code or zero; missing never matches'
        self.pred['encoding_scope']='main annual-relative or static dictionary; training-frozen overrides are given separately'
        self.rows=self.pred.set_index('predicate_id').to_dict('index')
        self.by_source={s:df.index.to_list() for s,df in self.pred.set_index('predicate_id').groupby('source_field')}
        self.pred.to_csv(TASK_O/'核查与数据表/19_新版完整谓词及逐年物理边界.csv',index=False,encoding='utf-8-sig')
        self.pred.to_csv((TASK_O/'PWD_新版谓词与逐年分箱_20261111.csv'),index=False,encoding='utf-8-sig')
    def risk(self,year,quality=True,unitmask=None):
        r=(self.shoufa==0)|(self.shoufa>=year)
        if quality:r&=year<self.censor
        if unitmask is not None:r&=unitmask
        return r
    def score_new(self,year,families,cuts=None,unitmask=None,quality=True,spatial=False):
        risk=self.risk(year,quality,unitmask);positive=risk&(self.shoufa==year)
        pi=np.flatnonzero(positive);total=int(risk.sum());positives=len(pi)
        ts={f:tuple(f.split(';')) for f in families};tokens=sorted({t for f in ts.values() for t in f})
        bits={};pb={}
        if spatial:
            provrows={p:np.flatnonzero(risk&(self.province==p)) for p in self.active_provinces}
            dense={p:{} for p in provrows}
        for token in tokens:
            item=self.item_for(token,year)
            if item is None:
                bits[token]=0;pb[token]=0
                if spatial:
                    for p in dense:dense[p][token]=0
                continue
            mask=self.match(item,cuts=cuts);pb[token]=packed_int(mask[pi]);mask&=risk;bits[token]=packed_int(mask)
            if spatial:
                for p,rr in provrows.items():dense[p][token]=packed_int(mask[rr])
        if spatial:
            pp={p:packed_int(self.province[pi]==p) for p in dense}
            pt={p:(len(provrows[p]),pp[p].bit_count()) for p in dense}
        result=[];geo=[];last=None;bb=bp=0;last3=None;b3=p3=0
        for f in sorted(families,key=lambda f:ts[f]):
            t=ts[f]
            if last!=t[:2]:last=t[:2];bb=bits[t[0]]&bits[t[1]];bp=pb[t[0]]&pb[t[1]];last3=None
            b=bb;p=bp
            if len(t)>=3:
                if last3!=t[:3]:last3=t[:3];b3=bb&bits[t[2]];p3=bp&pb[t[2]]
                b=b3;p=p3
            if len(t)==4:b&=bits[t[3]];p&=pb[t[3]]
            result.append(dict(rule_family=f,year=year,**metrics(b.bit_count(),p.bit_count(),total,positives)))
            if spatial:
                for prov,dbits in dense.items():
                    npop,npos=pt[prov]
                    if npop:
                        cover=dbits[t[0]]
                        for token in t[1:]:cover&=dbits[token]
                        geo.append(dict(rule_family=f,year=year,province=self.province_names[prov],**metrics(cover.bit_count(),(p&pp[prov]).bit_count(),npop,npos)))
        return pd.DataFrame(result),pd.DataFrame(geo)

def risk_manifest(store):
    rows=[];totals=[]
    for y in YEARS:
        for quality,name in [(True,'main_linkage_censored'),(False,'uncensored_sensitivity')]:
            risk=store.risk(y,quality);pos=risk&(store.shoufa==y)
            totals.append(dict(scope=name,year=y,risk_records=int(risk.sum()),positive_records=int(pos.sum())))
            for p,namep in enumerate(store.province_names):
                rows.append(dict(scope=name,year=y,province=namep,risk_records=int((risk&(store.province==p)).sum()),positive_records=int((pos&(store.province==p)).sum())))
    outcsv(pd.DataFrame(rows),'20_最终省年风险集与未截尾敏感性计数.csv');outcsv(pd.DataFrame(totals),'21_最终年度风险集验收计数.csv')
    a=pd.DataFrame(totals);main=a[a.scope=='main_linkage_censored'];alt=a[a.scope=='uncensored_sensitivity']
    keys=np.asarray(store.keys);order=np.load(OLD/'source_key_order.npy',mmap_mode='r')
    assert np.all(np.diff(keys[order])>0)
    assert np.isin(store.shoufa,[0,*range(2000,2022)]).all()
    assert all((store.shoufa[store.risk(y)]==0).all() or (store.shoufa[store.risk(y)]>=y).all() or np.all((store.shoufa[store.risk(y)]==0)|(store.shoufa[store.risk(y)]>=y)) for y in YEARS)
    summary=dict(source_polygon_records=len(keys),unique_source_keys=True,prebaseline_records=int(((store.shoufa>0)&(store.shoufa<2001)).sum()),
      main_risk_records=int(main.risk_records.sum()),main_events=int(main.positive_records.sum()),uncensored_risk_records=int(alt.risk_records.sum()),uncensored_events=int(alt.positive_records.sum()),
      units_with_linkage_censoring=int((store.censor<=2020).sum()),host_code_exclusions=0,
      host_definition='author-designated susceptible management units, unchanged',
      linkage_quality='15,574 potential-overlap units censored from first ambiguous-source year; sensitivity retains all years',
      first_event_rule='negative before first recorded year, positive once at first recorded year, excluded thereafter; pre-2001 events excluded throughout',
      dynamic_pressure='updated reference-point-based area-weighted spatial/time decay, explicitly distinguished from original polygon centroids',
      shared_boundary='affected-target complete topology recomputation, local AEQD, 1m grid; all values finite and nonnegative',
      caveat='no-overlap source events remain outside current harmonized footprint; no fabricated linkage or covariate assignment; geographic coverage limitations remain')
    js(TASK_W/'cache/final_dataset_manifest.json',summary);js(TASK_O/'数据集验收与研究范围说明.json',summary)
    log(f'Risk dataset checks complete: {summary}')

def transactions(store,name,years,seed=SEED,omit=(),unitmask=None,cuts=None,quality=True):
    cache=TASK_W/'cache'/f'{name}_bits.pkl'
    if cache.exists():
        with cache.open('rb') as f:return pickle.load(f)
    positives=[];negatives=[];py=[];ny=[];counts=[]
    for y in years:
        risk=store.risk(y,quality,unitmask);pos=risk&(store.shoufa==y);neg=risk&~pos
        h=splitmix64(store.keys^np.uint64(seed)^np.uint64(y*1000003))
        chosen=neg&(h%np.uint64(1000000)<5000)
        p=np.flatnonzero(pos);nn=np.flatnonzero(chosen);positives.append(p);negatives.append(nn)
        py.append(np.full(len(p),y,dtype=np.int16));ny.append(np.full(len(nn),y,dtype=np.int16))
        for prov,pname in enumerate(store.province_names):
            counts.append(dict(run=name,year=y,province=pname,eligible_positive_records=int((pos&(store.province==prov)).sum()),eligible_negative_records=int((neg&(store.province==prov)).sum()),sampled_positive_records=int((pos&(store.province==prov)).sum()),sampled_negative_records=int((chosen&(store.province==prov)).sum()),positive_inclusion_probability=1,negative_inclusion_probability=.005,seed=seed))
        log(f'{name}: sampled year {y}; positive={len(p):,}, negative={len(nn):,}')
    pp=np.concatenate(positives);nn=np.concatenate(negatives);rows=np.concatenate([pp,nn]);year=np.concatenate([*py,*ny]);nt=len(rows);np_=len(pp)
    assert np.all(store.shoufa[pp]==year[:np_])
    assert np.all((store.shoufa[nn]==0)|(store.shoufa[nn]>year[np_:]));assert np.all(year<store.censor[rows]) if quality else True
    np.savez_compressed(TASK_W/'cache'/f'{name}_indices.npz',source_row=rows,year=year,positive_prefix_count=np.array([np_]))
    outcsv(pd.DataFrame(counts),f'抽样_{name}_省年分母概率.csv')
    full={};positivebits={}
    for source,items in store.by_source.items():
        match=re.search(r'_(20\d{2})$',source)
        if match and int(match[1]) not in years:continue
        if all(store.rows[it]['family'] in omit for it in items):continue
        idx=np.flatnonzero(year==int(match[1])) if match else np.arange(nt)
        values=np.asarray(store.data[source][rows[idx]])
        for item in items:
            if store.rows[item]['family'] in omit:continue
            mask=np.zeros(nt,dtype=bool);mask[idx]=store.match(item,values=values,cuts=cuts)
            full[item]=packed_int(mask);positivebits[item]=packed_int(mask[:np_])
    obj=dict(total=nt,positive_count=np_,positive_bits=positivebits,full_bits=full,train_years=list(years),source=f'repaired author-defined cohort; run {name}; uniform negative record Bernoulli probability .005; seed {seed}',negative_probability=.005)
    with cache.open('wb') as f:pickle.dump(obj,f,protocol=5)
    return obj

def mine_run(store,name,years,**kwargs):
    return legacy_miner.mine(transactions(store,name,years,**kwargs),name)

def run(args):
    assert (TASK_W/'cache/boundary_full_complete.json').exists()
    assert (TASK_W/'cache/final_linkage_qa_complete.json').exists()
    store=RebuiltStore();risk_manifest(store)
    main=mine_run(store,'new_main',YEARS)
    summary=json.loads((TASK_W/'mining/new_main/summary.json').read_text(encoding='utf-8'))
    js(TASK_W/'cache/main_mining_summary.json',summary)
    main.to_csv(TASK_O/'核查与数据表/22_新主挖掘严格筛选规则.csv',index=False,encoding='utf-8-sig')
    log('New main mining complete')
    if args.main_only:return
    summaries=[]
    for k in [1,2]:
        name=f'new_repeat_{k}';mine_run(store,name,YEARS,seed=SEED+k*10007);summaries.append(json.loads((TASK_W/'mining'/name/'summary.json').read_text(encoding='utf-8')))
    for name,omit in [('new_no_roads',('Dis_rd','W_rd_den_5')),('new_no_GDP',('GDP',)),('new_no_roads_GDP',('Dis_rd','W_rd_den_5','GDP')),('new_no_DBH',('DBH',))]:
        mine_run(store,name,YEARS,omit=omit);summaries.append(json.loads((TASK_W/'mining'/name/'summary.json').read_text(encoding='utf-8')))
    mine_run(store,'new_uncensored',YEARS,quality=False)
    summaries.append(json.loads((TASK_W/'mining/new_uncensored/summary.json').read_text(encoding='utf-8')))
    for fold,held in FOLDS.items():
        name=f'new_temporal_{fold}';mine_run(store,name,sorted(set(YEARS)-set(held)),seed=SEED+fold*100003)
        summaries.append(json.loads((TASK_W/'mining'/name/'summary.json').read_text(encoding='utf-8')))
    # County groups are allocated before any discovery in a spatial run.
    spatialgroup=(splitmix64(np.asarray(store.county,dtype=np.uint64)^np.uint64(SEED))%np.uint64(5)).astype(np.uint8)
    np.save(TASK_W/'cache/spatial_county_fold.npy',spatialgroup)
    for fold in range(5):
        name=f'new_spatial_{fold+1}';mine_run(store,name,YEARS,unitmask=spatialgroup!=fold,seed=SEED+(fold+1)*300007)
        summaries.append(json.loads((TASK_W/'mining'/name/'summary.json').read_text(encoding='utf-8')))
    for window,end in enumerate([2005,2010,2015],1):
        name=f'new_forward_{window}';mine_run(store,name,[y for y in YEARS if y<=end],seed=SEED+window*700001)
        summaries.append(json.loads((TASK_W/'mining'/name/'summary.json').read_text(encoding='utf-8')))
    # Absolute annual-variable thresholds are frozen using training-year values only.
    cutrows=[]
    for fold,held in FOLDS.items():
        train=sorted(set(YEARS)-set(held));cuts,rr=store.cuts(train,f'new_fixed_{fold}');cutrows+=rr
        js(TASK_W/'cache'/f'fixed_cuts_{fold}.json',cuts)
        name=f'new_fixed_{fold}';mine_run(store,name,train,seed=SEED+fold*100003,cuts=cuts)
        summaries.append(json.loads((TASK_W/'mining'/name/'summary.json').read_text(encoding='utf-8')))
    outcsv(pd.DataFrame(cutrows),'23_训练期冻结物理阈值.csv')
    js(TASK_W/'cache/all_discovery_runs_complete.json',dict(runs=summaries))
    log('All planned discovery, repeated-sampling, ablation, county and temporal runs completed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset-and-main',action='store_true');p.add_argument('--main-only',action='store_true');args=p.parse_args();run(args)
