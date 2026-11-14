from env import *
import time,math,gc,hashlib
from new_analysis import RebuiltStore,mine_run,YEARS,FOLDS
from common import metrics
def readrules(name):return pd.read_csv(W/'mining'/name/'strict_rules.csv',encoding='utf-8-sig')
def waitfile(path):
    while not Path(path).exists():time.sleep(5)
def pool(d):
    a=int(d.positive_support_count.sum());n=int(d.antecedent_support_count.sum());nt=int(d.risk_total.sum());np_=int(d.risk_positive_total.sum())
    return metrics(n,a,nt,np_)
def run():
    waitfile(W/'cache/main_mining_summary.json')
    store=RebuiltStore();main=readrules('new_main')
    canonical=main.sort_values('secondary_score',ascending=False).drop_duplicates('rule_family').copy()
    families=sorted(canonical.rule_family);assert len(families)>0
    allannual=[];geoacc=None
    for year in YEARS:
        checkpoint=W/'cache'/f'annual_full_score_{year}.pkl'
        if checkpoint.exists():d,g=pd.read_pickle(checkpoint)
        else:
            d,g=store.score_new(year,families,spatial=True)
            pd.to_pickle((d,g),checkpoint)
        allannual.append(d)
        g['expected_events_year_standardized']=g.antecedent_support_count*g.risk_positive_total/g.risk_total
        fields=['antecedent_support_count','positive_support_count','risk_total','risk_positive_total','expected_events_year_standardized']
        inc=g.set_index(['rule_family','province'])[fields]
        geoacc=inc if geoacc is None else geoacc.add(inc,fill_value=0)
        log(f'Validation: full-risk annual and provincial counts {year} completed')
        del g;gc.collect()
    annual=pd.concat(allannual,ignore_index=True)
    annual.to_csv(O/'核查与数据表/24_新主规则族完整逐年指标.csv',index=False,encoding='utf-8-sig')
    geo=geoacc.reset_index();geo['confidence']=geo.positive_support_count/geo.antecedent_support_count.replace(0,np.nan)
    geo['base_rate']=geo.risk_positive_total/geo.risk_total
    geo['lift']=geo.confidence/geo.base_rate.replace(0,np.nan)
    a=geo.positive_support_count;n=geo.antecedent_support_count;nt=geo.risk_total;p=geo.risk_positive_total
    geo['odds_ratio']=((a+.5)*(nt-n-p+a+.5))/((n-a+.5)*(p-a+.5))
    geo['year_standardized_enrichment']=a/geo.expected_events_year_standardized.replace(0,np.nan)
    geo['province_pass']=(a>=20)&(n>=1000)&(geo.lift>=2)&(geo.odds_ratio>=2)
    geo.to_csv(O/'核查与数据表/25_新主规则族逐省汇总验证.csv',index=False,encoding='utf-8-sig')
    summaries=[]
    for family,d in annual.groupby('rule_family'):
        e=d[(d.antecedent_support_count>=1000)&(d.antecedent_support_count*d.base_rate>=5)]
        strong=(e.lift>=1.5)&(e.positive_support_count>=10);direction=e.lift>1
        gg=geo[geo.rule_family==family];positive=gg.positive_support_count.sum()
        summaries.append(dict(rule_family=family,**pool(d),evaluable_years=len(e),evaluable_five_year_periods=int(((e.year-2001)//5).nunique()),annual_strong_fraction=float(strong.mean()) if len(e) else 0,annual_direction_fraction=float(direction.mean()) if len(e) else 0,annual_lift_min=float(e.lift.min()) if len(e) else np.nan,annual_lift_max=float(e.lift.max()) if len(e) else np.nan,annual_lift_median=float(e.lift.median()) if len(e) else np.nan,province_pass_count=int(gg.province_pass.sum()),largest_province_positive_share=float(gg.positive_support_count.max()/positive) if positive else np.nan,largest_province=str(gg.loc[gg.positive_support_count.idxmax(),'province']) if len(gg) else ''))
    summary=pd.DataFrame(summaries)
    summary['spatial_scope']=np.where((summary.province_pass_count>=3)&(summary.largest_province_positive_share<=.75),'broad_multi_province','spatially_concentrated')
    foldrows=[];mapping=[]
    for fold,held in FOLDS.items():
        waitfile(W/'mining'/f'new_temporal_{fold}'/'summary.json')
        trainrules=readrules(f'new_temporal_{fold}');found=set(trainrules.rule_family)
        trainrules[['antecedent','rule_family']].assign(fold=fold,train_years=';'.join(map(str,sorted(set(YEARS)-set(held))))).to_csv(O/'核查与数据表'/f'时间折{fold}_原规则到规则族映射.csv',index=False,encoding='utf-8-sig')
        for family,d in annual[annual.year.isin(held)].groupby('rule_family'):
            mm=pool(d);expected=float((d.antecedent_support_count*d.base_rate).sum())
            foldrows.append(dict(rule_family=family,fold=fold,training_discovered=family in found,train_years=';'.join(map(str,sorted(set(YEARS)-set(held)))),validation_years=';'.join(map(str,held)),**mm,year_standardized_enrichment=mm['positive_support_count']/expected if expected else np.nan))
        log(f'Validation: relative annual-encoding time fold {fold} discovery/evaluation joined')
    folds=pd.DataFrame(foldrows)
    folds['strong_pass']=(folds.antecedent_support_count>=1000)&(folds.positive_support_count>=10)&(folds.lift>=1.5)
    folds['direction_pass']=(folds.antecedent_support_count>=1000)&(folds.positive_support_count>=10)&(folds.lift>1)
    folds['joint_strong']=folds.training_discovered&folds.strong_pass;folds['joint_direction']=folds.training_discovered&folds.direction_pass
    folds.to_csv(O/'核查与数据表/26_逐规则族留年训练与验证完整指标.csv',index=False,encoding='utf-8-sig')
    agg=folds.groupby('rule_family')[['training_discovered','strong_pass','direction_pass','joint_strong','joint_direction']].sum().add_suffix('_folds').reset_index()
    summary=summary.merge(agg,on='rule_family',validate='1:1')
    enough=(summary.evaluable_years>=10)&(summary.evaluable_five_year_periods>=3)
    strong=enough&(summary.annual_strong_fraction>=.8)&(summary.joint_strong_folds>=4)
    direction=enough&(summary.annual_direction_fraction>=.8)&(summary.joint_direction_folds>=4)
    summary['temporal_class']=np.select([strong,direction,~enough],['strong_internal_temporal','direction_consistent','insufficient_coverage'],default='not_persistent')
    summary=summary.merge(canonical[['rule_family','antecedent','length','secondary_score','log10_p','log10_q']],on='rule_family',validate='1:1')
    summary['family_id']=summary.rule_family.map(lambda s:'F'+hashlib.sha256(s.encode()).hexdigest()[:12]);assert summary.family_id.nunique()==len(summary)
    summary=summary.sort_values(['temporal_class','secondary_score'],ascending=[True,False])
    summary.to_csv(O/'核查与数据表/27_新规则族时间与空间分类主结果.csv',index=False,encoding='utf-8-sig')
    early=dict(main_strict_rule_versions=len(main),main_unique_families=len(summary),temporal_counts=summary.temporal_class.value_counts().to_dict(),spatial_temporal_crosstab=pd.crosstab(summary.spatial_scope,summary.temporal_class).to_dict(),interpretation='internal robustness after full-data selection; not independent external validation')
    js(W/'cache/temporal_summary.json',early);js(O/'新时间稳定性主结果摘要.json',early)
    log(f'Primary temporal result saved: {early}')
    # Continue absolute-threshold, repeated sampling, ablation and chronological checks after rediscovery completes.
    waitfile(W/'cache/all_discovery_runs_complete.json')
    if not (W/'mining/new_uncensored/summary.json').exists():mine_run(store,'new_uncensored',YEARS,quality=False)
    comparison=[];mainset=set(canonical.rule_family)
    for name in ['new_repeat_1','new_repeat_2','new_no_roads','new_no_GDP','new_no_roads_GDP','new_no_DBH','new_uncensored']:
        rr=readrules(name);cc=rr.drop_duplicates('rule_family');ss=set(cc.rule_family)
        comparison.append(dict(run=name,strict_rule_versions=len(rr),families=len(ss),common_with_main=len(ss&mainset),jaccard=len(ss&mainset)/len(ss|mainset) if ss|mainset else 1,top100_jaccard=len(set(cc.head(100).rule_family)&set(canonical.head(100).rule_family))/len(set(cc.head(100).rule_family)|set(canonical.head(100).rule_family))))
    outcsv(pd.DataFrame(comparison),'28_重复抽样删变量与对应不确定性重挖比较.csv')
    fixedrows=[]
    for fold,held in FOLDS.items():
        cuts=json.loads((W/'cache'/f'fixed_cuts_{fold}.json').read_text(encoding='utf-8'));found=set(readrules(f'new_fixed_{fold}').rule_family)
        dd=[]
        for y in held:
            d,_=store.score_new(y,families,cuts=cuts);dd.append(d)
        dd=pd.concat(dd,ignore_index=True)
        for family,d in dd.groupby('rule_family'):
            mm=pool(d);fixedrows.append(dict(rule_family=family,fold=fold,training_discovered=family in found,**mm))
        log(f'Validation: training-frozen numeric thresholds fold {fold} scored')
    fixed=pd.DataFrame(fixedrows);fixed['joint_strong']=fixed.training_discovered&(fixed.antecedent_support_count>=1000)&(fixed.positive_support_count>=10)&(fixed.lift>=1.5)
    fixed.to_csv(O/'核查与数据表/29_训练冻结物理阈值留年验证.csv',index=False,encoding='utf-8-sig')
    summary=summary.merge(fixed.groupby('rule_family').joint_strong.sum().rename('fixed_joint_strong_folds'),on='rule_family')
    forward=[]
    for window,end in enumerate([2005,2010,2015],1):
        held=list(range(end+1,end+6));found=set(readrules(f'new_forward_{window}').rule_family)
        for family,d in annual[annual.year.isin(held)].groupby('rule_family'):
            mm=pool(d);forward.append(dict(rule_family=family,window=window,training_end=end,validation_years=';'.join(map(str,held)),training_discovered=family in found,**mm))
    forward=pd.DataFrame(forward);forward['joint_strong']=forward.training_discovered&(forward.antecedent_support_count>=1000)&(forward.positive_support_count>=10)&(forward.lift>=1.5)
    outcsv(forward,'30_三个向前时间窗口验证.csv')
    summary=summary.merge(forward.groupby('rule_family').joint_strong.sum().rename('forward_joint_strong_windows'),on='rule_family')
    selected=summary[summary.temporal_class.isin(['strong_internal_temporal','direction_consistent'])].rule_family.tolist()
    spatialrows=[];groups=np.load(W/'cache/spatial_county_fold.npy',mmap_mode='r')
    for fold in range(5):
        found=set(readrules(f'new_spatial_{fold+1}').rule_family);dd=[]
        for y in YEARS:
            d,_=store.score_new(y,selected,unitmask=groups==fold);dd.append(d)
        dd=pd.concat(dd,ignore_index=True)
        for family,d in dd.groupby('rule_family'):
            mm=pool(d);spatialrows.append(dict(rule_family=family,fold=fold+1,training_discovered=family in found,**mm))
        log(f'Validation: independent county-label holdout fold {fold+1} scored')
    spatial=pd.DataFrame(spatialrows)
    if len(spatial):
        spatial['joint_strong']=spatial.training_discovered&(spatial.antecedent_support_count>=1000)&(spatial.positive_support_count>=10)&(spatial.lift>=1.5)
        summary=summary.merge(spatial.groupby('rule_family').joint_strong.sum().rename('county_joint_strong_folds'),on='rule_family',how='left')
    outcsv(spatial,'31_先划分县组再发现的留出验证.csv')
    summary.to_csv(O/'核查与数据表/32_完整验证后的新规则族结果.csv',index=False,encoding='utf-8-sig')
    js(W/'cache/validation_complete.json',dict(families=len(summary),primary=early,absolute_threshold_strong_families=int((summary.fixed_joint_strong_folds>=4).sum()),discovery_sensitivity_runs=len(comparison),spatially_tested_temporal_families=len(selected)))
    log('All validation stages complete')
if __name__=='__main__':run()
