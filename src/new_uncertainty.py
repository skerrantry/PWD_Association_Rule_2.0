from env import *
from new_analysis import RebuiltStore,YEARS
def run():
    store=RebuiltStore();d=pd.read_csv(O/'核查与数据表/32_完整验证后的新规则族结果.csv')
    candidates=d[d.temporal_class=='strong_internal_temporal'].sort_values('secondary_score',ascending=False)
    chosen=pd.concat([candidates[candidates.spatial_scope==scope].head(10) for scope in ['broad_multi_province','spatially_concentrated']]).drop_duplicates('rule_family')
    k=len(store.groups);r=np.zeros((20,k));p=np.zeros((20,k))
    for yi,y in enumerate(YEARS):
        risk=store.risk(y);pos=risk&(store.shoufa==y)
        r[yi]=np.bincount(store.county[risk],minlength=k);p[yi]=np.bincount(store.county[pos],minlength=k)
    rng=np.random.default_rng(20261111);b=500;weights=np.zeros((b,k),dtype=np.float64);groups=[]
    for name,g in store.groups.groupby('province_group'):
        ids=g.county_group_id.to_numpy(dtype=int);groups.append(ids)
        weights[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=b)
    rb=weights@r.sum(axis=0);pb=weights@p.sum(axis=0);results=[]
    for _,row in chosen.iterrows():
        family=row.rule_family;n=np.zeros((20,k));a=np.zeros((20,k))
        for yi,y in enumerate(YEARS):
            mask=store.risk(y)
            for token in family.split(';'):mask&=store.match(store.item_for(token,y))
            n[yi]=np.bincount(store.county[mask],minlength=k)
            a[yi]=np.bincount(store.county[mask&(store.shoufa==y)],minlength=k)
        assert int(n.sum())==int(row.antecedent_support_count)
        assert int(a.sum())==int(row.positive_support_count)
        nb=weights@n.sum(axis=0);ab=weights@a.sum(axis=0)
        with np.errstate(divide='ignore',invalid='ignore'):
            lifts=(ab/nb)/(pb/rb)
            odds=(ab+.5)*(rb-nb-pb+ab+.5)/((nb-ab+.5)*(pb-ab+.5))
        expected=np.zeros(b);e0=0.
        for ids in groups:
            nn=weights[:,ids]@n[:,ids].T;pp=weights[:,ids]@p[:,ids].T;rr=weights[:,ids]@r[:,ids].T
            expected+=np.sum(nn*np.divide(pp,rr,out=np.zeros_like(pp),where=rr>0),axis=1)
            rp=r[:,ids].sum(axis=1);pp0=p[:,ids].sum(axis=1);nn0=n[:,ids].sum(axis=1)
            e0+=float(np.sum(nn0*np.divide(pp0,rp,out=np.zeros_like(pp0),where=rp>0)))
        std=np.divide(ab,expected,out=np.full(b,np.nan),where=expected>0)
        ci=lambda x:np.nanquantile(x,[.025,.975]).tolist()
        results.append(dict(rule_family=family,family_id=row.family_id,spatial_scope=row.spatial_scope,antecedent_support_count=int(n.sum()),positive_support_count=int(a.sum()),lift=float(row.lift),lift_ci_lower=ci(lifts)[0],lift_ci_upper=ci(lifts)[1],odds_ratio=float(row.odds_ratio),or_ci_lower=ci(odds)[0],or_ci_upper=ci(odds)[1],province_year_standardized_enrichment=float(a.sum()/e0) if e0 else np.nan,standardized_ci_lower=ci(std)[0],standardized_ci_upper=ci(std)[1],bootstrap_repetitions=b,finite_lift_repetitions=int(np.isfinite(lifts).sum()),finite_standardized_repetitions=int(np.isfinite(std).sum()),grouping='original county-code groups, resampled with replacement within province',interval='percentile 95%; conditional on selected family and archived data'))
        outcsv(pd.DataFrame(results),'35_代表规则族省内县组bootstrap区间.csv')
        log(f'Grouped uncertainty completed: {family}')
    geo=pd.read_csv(O/'核查与数据表/25_新主规则族逐省汇总验证.csv')
    expected=geo.groupby('rule_family').expected_events_year_standardized.sum()
    adjusted=d[['family_id','rule_family','lift','positive_support_count','temporal_class','spatial_scope']].merge(expected,on='rule_family')
    adjusted['province_year_standardized_enrichment']=adjusted.positive_support_count/adjusted.expected_events_year_standardized.replace(0,np.nan)
    outcsv(adjusted,'36_全部规则族省年基线标准化敏感性.csv')
    js(W/'cache/uncertainty_complete.json',dict(examples=len(results),bootstrap_repetitions=b,interval_method='province-stratified original-county-group percentile bootstrap',all_family_standardization=True,limitation='conditional intervals; selection and cross-county dependence not eliminated'))
if __name__=='__main__':run()
