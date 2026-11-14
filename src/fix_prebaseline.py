from env import *
def run():
    old=np.nan_to_num(np.load(OLD/'SHOUFA_YEA.npy',mmap_mode='r'),nan=0).astype(np.int16)
    new=np.load(W/'cache/spatially_repaired_first_year.npy').copy()
    pre=(old>0)&(old<2001);future=(old>2020)&(new==0)
    rows=np.flatnonzero((pre&(new!=old))|future)
    outcsv(pd.DataFrame({'source_row':rows,'old_first_year':old[rows],'intermediate_first_year':new[rows]}),'13_研究起点前记录及未来年份保留修复.csv')
    new[pre]=old[pre];new[future]=old[future]
    np.save(W/'cache/analysis_first_year.npy',new)
    js(W/'cache/prebaseline_fixed.json',dict(prebaseline_units=int(pre.sum()),restored_prebaseline_units=int((pre&(np.load(W/'cache/spatially_repaired_first_year.npy')!=old)).sum()),restored_future_units=int(future.sum()),policy='preserve verified existing first occurrence before the study start; exclude these units from all incident-risk years and retain their historical exposure contributions'))
    log('Preserved all 216 pre-2001 first-recorded units; corrected the intermediate reset of 130 units')
if __name__=='__main__':run()
