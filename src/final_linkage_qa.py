from env import *
import sqlite3,pyogrio,shapely,gc
from collections import defaultdict
def run():
    db=sqlite3.connect(W/'cache/geometry_bbox.sqlite')
    keys=np.load(OLD/'source_key.npy',mmap_mode='r');order=np.load(OLD/'source_key_order.npy',mmap_mode='r');sk=keys[order]
    partition=(keys>>np.uint64(32)).astype(np.uint8)
    label=np.load(W/'cache/analysis_first_year.npy').copy();n=len(label)
    m=pd.read_csv(O/'核查与数据表/04_源FID到分析小班修复映射.csv');m2=pd.read_csv(O/'核查与数据表/06_第二轮多边形重叠匹配明细.csv')
    resolved=set(m.loc[m.target_source_row>=0,'source_fid'])|set(m2.source_fid)
    remaining=m[~m.source_fid.isin(resolved)].copy()
    censor=np.full(n,2021,dtype=np.int16);rows=[];newlinks=[];cached={}
    for at in range(0,len(remaining),500):
        chunk=remaining.iloc[at:at+500]
        mm,ff,gg,aa=pyogrio.raw.read(str(F/'80_Infectedxiaoban_2020/merged.shp'),columns=[],fids=chunk.source_fid.astype(int).tolist(),return_fids=True)
        sources={int(fid):shapely.make_valid(shapely.from_wkb(g)) for fid,g in zip(ff,gg)}
        for _,r in chunk.iterrows():
            s=sources[int(r.source_fid)];year=int(r.source_year)
            if s.is_empty or s.area<=0:
                rows.append(dict(source_fid=int(r.source_fid),province=int(r.province),source_year=year,status='invalid_or_empty_source_geometry',intersecting_targets=0,accepted_targets=0));continue
            b=s.bounds
            targets=np.array([z[0] for z in db.execute('SELECT source_row FROM bbox WHERE minx<=? AND maxx>=? AND miny<=? AND maxy>=?',(b[2],b[0],b[3],b[1]))],dtype=np.int64)
            missing=np.array([int(t) for t in targets if int(t) not in cached],dtype=np.int64)
            for tp in np.unique(partition[missing]):
                rr=missing[partition[missing]==tp];ids=(keys[rr]&np.uint64(0xffffffff)).astype(np.int64)
                mt,tf,tg,ta=pyogrio.raw.read(str(F/'Sub_Pine_All.gdb'),layer=f'T{tp}',columns=[],fids=ids.tolist(),return_fids=True)
                for fid,g in zip(tf,tg):
                    k=(np.uint64(tp)<<np.uint64(32))|np.uint64(fid);loc=int(order[np.searchsorted(sk,k)])
                    cached[loc]=shapely.make_valid(shapely.from_wkb(g))
            overlaps=[];accepted=[]
            for loc in targets:
                t=cached[int(loc)]
                if t.is_empty or t.area<=0 or not s.intersects(t):continue
                a=s.intersection(t).area
                if a<=1e-12:continue
                fs=a/s.area;ft=a/t.area;overlaps.append(int(loc))
                if (fs>=.5 and ft>=.1) or (ft>=.5 and fs>=.1):accepted.append((int(loc),fs,ft))
            if accepted:
                status='resolved_by_complete_polygon_index'
                for loc,fs,ft in accepted:
                    label[loc]=min(int(label[loc]),year) if label[loc]>0 else year
                    newlinks.append(dict(source_fid=int(r.source_fid),source_year=year,province=int(r.province),target_source_row=loc,source_overlap_fraction=fs,target_overlap_fraction=ft))
            elif overlaps:
                status='partial_overlap_ambiguous_linkage'
                for loc in overlaps:censor[loc]=min(int(censor[loc]),year)
            else:status='no_polygon_area_overlap_with_analysis_footprint'
            rows.append(dict(source_fid=int(r.source_fid),province=int(r.province),source_year=year,status=status,intersecting_targets=len(overlaps),accepted_targets=len(accepted)))
            if len(cached)>70000:cached.clear()
        outcsv(pd.DataFrame(rows),'14_未匹配源记录完整空间索引终审.csv')
        if newlinks:outcsv(pd.DataFrame(newlinks),'15_完整空间索引新增对应关系.csv')
        log(f'Final linkage spatial QA {min(at+500,len(remaining)):,}/{len(remaining):,}')
    old_intermediate=np.load(W/'cache/analysis_first_year.npy')
    np.save(W/'cache/analysis_first_year.npy',label);np.save(W/'cache/linkage_censor_year.npy',censor)
    ambiguous=np.flatnonzero(censor<=2020)
    outcsv(pd.DataFrame({'source_row':ambiguous,'first_linkage_ambiguity_year':censor[ambiguous],'first_confirmed_year':label[ambiguous]}),'16_对应关系不确定小班的保守截尾年份.csv')
    df=pd.DataFrame(rows)
    summary=dict(remaining_source_records_examined=len(df),status_counts=df.status.value_counts().to_dict(),new_source_records_resolved=int((df.status=='resolved_by_complete_polygon_index').sum()),new_target_links=len(newlinks),analysis_units_with_linkage_censoring=len(ambiguous),first_year_changed_in_this_stage=int((label!=old_intermediate).sum()),
      host_eligibility='unchanged forestry-department designated population; no species-code screening',
      main_risk_rule='include unit-year if no confirmed earlier event and year precedes any unresolved overlapping source-event year; positive event only at confirmed first year before linkage censoring',
      censoring_note='quality-based conservative censoring, not a claim of disease absence or a redefinition of host susceptibility; uncensored full-cohort sensitivity required',
      outside_note='source polygons without any area overlap are outside the harmonized analysis footprint; report separately rather than fabricate links or add unsupported covariates')
    js(W/'cache/final_linkage_qa_complete.json',summary);js(O/'数据对应关系终审摘要.json',summary)
    log(f'Final linkage QA complete: {summary}')
if __name__=='__main__':run()
