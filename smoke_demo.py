from pathlib import Path
import tempfile,json,os,sys
root=Path(__file__).resolve().parent
tmp=Path(tempfile.mkdtemp(prefix='pwd-demo-'))
config={'raw_backup_root':str(tmp/'raw'),'raw_csv':str(tmp/'raw/raw.csv'),'numeric_cache':str(tmp/'cache'),'work_dir':str(tmp/'work'),'output_dir':str(tmp/'out'),'source_units':10000}
(tmp/'config.json').write_text(json.dumps(config),encoding='utf-8');os.environ['PWD_CONFIG']=str(tmp/'config.json');sys.path.insert(0,str(root/'src'))
import numpy as np
from common import packed_int,family_key,metrics
from predicates import splitmix64
from miner import mine,exact_log_p
from scipy.stats import fisher_exact
N=10000;P=1000
mask_a=np.zeros(N,bool);mask_a[:600]=True;mask_a[1000:1600]=True
mask_b=np.zeros(N,bool);mask_b[:700]=True;mask_b[1000:1800]=True
obj={'total':N,'positive_count':P,'positive_bits':{'Area_Q5':packed_int(mask_a[:P]),'GDP_2018_Q5':packed_int(mask_b[:P])},'full_bits':{'Area_Q5':packed_int(mask_a),'GDP_2018_Q5':packed_int(mask_b)},'train_years':[2018],'source':'synthetic demonstration only'}
result=mine(obj,'synthetic');assert len(result)==1
r=result.iloc[0];assert r.antecedent_support_count==1200 and r.positive_support_count==600
assert r.rule_family=='Area_Q5;GDP_Q5'
assert np.isclose(r.lift,5)
a=np.array([6]);n=np.array([12]);lp=exact_log_p(a,n,100,10)
ref=fisher_exact([[6,6],[4,84]],alternative='greater').pvalue
assert np.isclose(np.exp(lp[0]),ref)
assert family_key(['GDP_2001_Q5','Area_Q5'])==family_key(['Area_Q5','GDP_2018_Q5'])
keys=np.arange(10000,dtype=np.uint64);h=splitmix64(keys^np.uint64(20261111));assert np.array_equal(h,splitmix64(keys^np.uint64(20261111)))
print('PASS: sampled discovery, exact count recovery, family mapping, effect denominator, Fisher upper tail and repeatable hash. Synthetic outputs:',tmp/'work/mining/synthetic')
