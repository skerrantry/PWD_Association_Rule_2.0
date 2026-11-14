from common import *
import argparse,itertools,pickle,zlib
from collections import Counter
from scipy.stats import hypergeom

def save_pickle(path,obj):
    with Path(path).open('wb') as f:pickle.dump(obj,f,protocol=5)

def load_old_fold(fold):
    path=A/f'09_temporal_remining_full_experiment/01_fold_transactions/fold_{fold}_train_transactions.csv'
    cache=C/f'fold_{fold}_transaction_bits.pkl'
    meta_path=C/f'fold_{fold}_transaction_bits.json'
    if cache.exists() and meta_path.exists() and json.loads(meta_path.read_text())['source']==source_metadata(path):
        with cache.open('rb') as f:return pickle.load(f)
    summary=readcsv(A/'09_temporal_remining_full_experiment/stage9_temporal_remining_full_experiment_summary.csv').set_index('key').value
    nt=int(summary[f'fold_{fold}_transactions']);np_=int(summary[f'fold_{fold}_positive_transactions'])
    full={};positive={};pidx=0;year_counts=Counter(); t=time.time()
    valid=set(YEARS)-set(FOLDS[fold]); annual_items={}
    with path.open(encoding='utf-8-sig',newline='') as f:
      rr=csv.reader(f); header=next(rr)
      assert header==['transaction_id','sample_id','province_fc','year','label','fold','items']
      for tid,row in enumerate(rr):
        y=int(row[3]);label=int(row[4]);assert y in valid
        byte=tid>>3; bit=1<<(tid&7)
        pbyte=pidx>>3;pbit=1<<(pidx&7)
        for item in row[6].split(';'):
          if item not in full:
            full[item]=bytearray((nt+7)//8);positive[item]=bytearray((np_+7)//8)
            match=re.search(r'_(20\d{2})_',item);annual_items[item]=int(match[1]) if match else 0
          assert annual_items[item] in (0,y)
          full[item][byte]|=bit
          if label:positive[item][pbyte]|=pbit
        pidx+=label;year_counts[(y,label)]+=1
        if (tid+1)%300000==0:log('mining',f'fold {fold}: loaded {tid+1:,}/{nt:,} transactions')
    assert tid+1==nt and pidx==np_
    obj=dict(total=nt,positive_count=np_,
      positive_bits={k:int.from_bytes(v,'little') for k,v in positive.items()},
      full_bits={k:int.from_bytes(v,'little') for k,v in full.items()},
      year_counts={f'{y}_{l}':n for (y,l),n in year_counts.items()},
      train_years=sorted(valid),source=str(path))
    save_pickle(cache,obj);save_json(meta_path,dict(source=source_metadata(path),load_seconds=time.time()-t))
    return obj

def exact_log_p(a,n,nt,np_):
    p=hypergeom.sf(a-1,nt,np_,n)
    lp=np.log(p,where=p>0,out=np.full(len(p),-np.inf))
    idx=np.flatnonzero(p==0)
    if len(idx):
      k=n[idx].astype(float);x=a[idx].astype(float)
      base=hypergeom.logpmf(x,nt,np_,k)
      term=np.ones(len(idx));s=np.ones(len(idx));active=np.ones(len(idx),bool)
      for j in range(10000):
        ratio=np.maximum(0,(np_-x)*(k-x)/((x+1)*(nt-np_-k+x+1)))
        term[active]*=ratio[active];s[active]+=term[active];x+=1
        next_ratio=np.maximum(0,(np_-x)*(k-x)/((x+1)*(nt-np_-k+x+1)))
        rem=np.where(next_ratio<1,term*next_ratio/np.maximum(1-next_ratio,1e-300),np.inf)
        active&=rem>s*1e-13
        if not active.any():break
      else:raise RuntimeError('Hypergeometric tail did not converge')
      lp[idx]=base+np.log(s)
    assert np.isfinite(lp).all()
    return lp

def mine(obj,name):
    out=W/'mining'/name;out.mkdir(parents=True,exist_ok=True)
    done=out/'summary.json'
    if done.exists() and (out/'strict_rules.csv').exists():
      log('mining',f'{name}: reuse completed exact mining');return readcsv(out/'strict_rules.csv')
    t=time.time();nt=obj['total'];np_=obj['positive_count']
    items=sorted(k for k,v in obj['positive_bits'].items() if v.bit_count()>=100)
    nodes=[(i,obj['positive_bits'][s],obj['full_bits'][s]) for i,s in enumerate(items)]
    combos=[];positives=[];coverage=[]
    def dfs(prefix,nodes):
      for i,(item,pbits,allbits) in enumerate(nodes):
        combo=prefix+(item,)
        if len(combo)>=2:
          combos.append(combo);positives.append(pbits.bit_count());coverage.append(allbits.bit_count())
        if len(combo)<4:
          children=[]
          for item2,p2,t2 in nodes[i+1:]:
            pp=pbits&p2
            if pp.bit_count()>=100:children.append((item2,pp,allbits&t2))
          if children:dfs(combo,children)
        if not prefix and i%30==0:log('mining',f'{name}: prefix {i+1}/{len(nodes)}, candidates={len(combos):,}')
    dfs((),nodes)
    a=np.asarray(positives,dtype=np.int64);n=np.asarray(coverage,dtype=np.int64)
    assert (n>=a).all() and (n<=nt).all()
    log('mining',f'{name}: {len(a):,} exact candidates; scoring Fisher/BH')
    confidence=a/n;base=np_/nt;lift=confidence/base
    odds=((a+.5)*(nt-n-np_+a+.5))/((n-a+.5)*(np_-a+.5))
    wracc=a/nt-n/nt*base
    lp=exact_log_p(a,n,nt,np_)
    order=np.argsort(lp,kind='stable');raw=lp[order]+math.log(len(a))-np.log(np.arange(1,len(a)+1))
    lq=np.empty(len(a));lq[order]=np.minimum(0,np.minimum.accumulate(raw[::-1])[::-1])
    keep=(a>=300)&(n>=1000)&(lift>=3)&(odds>=3)&(lq<math.log(.01))
    lookup={v:i for i,v in enumerate(combos) if len(v)<=3}
    labels=readcsv(A/'02_predicates/predicate_definitions_draft.csv').set_index('predicate_id').label.to_dict()
    rows=[];dropped=0
    for i in np.flatnonzero(keep):
      combo=combos[i];lg=wg=float('nan');best=''
      if len(combo)==4:
        subs=[lookup[s] for s in itertools.combinations(combo,3)]
        lg=float(lift[i]/max(lift[subs]));wg=float(wracc[i]/max(wracc[subs]))
        if not(lg>=1.2 or wg>=1.2):dropped+=1;continue
        bestidx=max(subs,key=lambda j:(lift[j],wracc[j]));best=';'.join(items[v] for v in combos[bestidx])
      strings=[items[j] for j in combo]
      simplicity={2:1.,3:.9,4:.72}[len(combo)]
      score=math.log1p(int(a[i]))*math.log(lift[i])*math.log(odds[i])*simplicity*(1+math.sqrt(max(0,wracc[i])))
      rows.append(dict(antecedent=';'.join(strings),antecedent_labels=' | '.join(labels.get(s,s) for s in strings),
        rule_family=family_key(strings),length=len(combo),sample_total_transactions=nt,sample_positive_transactions=np_,
        antecedent_support_count=int(n[i]),positive_support_count=int(a[i]),confidence=float(confidence[i]),
        base_rate=base,lift=float(lift[i]),odds_ratio=float(odds[i]),wracc=float(wracc[i]),
        log10_p=float(lp[i]/math.log(10)),log10_q=float(lq[i]/math.log(10)),secondary_score=score,
        best_subrule=best,lift_gain_ratio=lg,wracc_gain_ratio=wg))
    fields=['antecedent','antecedent_labels','rule_family','length','sample_total_transactions','sample_positive_transactions',
      'antecedent_support_count','positive_support_count','confidence','base_rate','lift','odds_ratio','wracc','log10_p','log10_q',
      'secondary_score','best_subrule','lift_gain_ratio','wracc_gain_ratio']
    strict=pd.DataFrame(rows,columns=fields).sort_values(['secondary_score','lift','positive_support_count'],ascending=False)
    strict.to_csv(out/'strict_rules.csv',index=False,encoding='utf-8-sig')
    padded=np.full((len(combos),4),-1,dtype=np.int16)
    for i,c in enumerate(combos):padded[i,:len(c)]=c
    np.savez_compressed(out/'candidate_counts_and_log_tests.npz',items=np.asarray(items),combos=padded,
      positive_support=a,antecedent_support=n,log_p=lp,log_q=lq)
    summary=dict(name=name,train_years=obj['train_years'],transactions=nt,positive_transactions=np_,
      frequent_items=len(items),candidates=len(a),strict_rules=len(strict),strict_families=int(strict.rule_family.nunique()),
      dropped_length4=dropped,elapsed_seconds=time.time()-t,source=obj.get('source','numeric cache'),
      fisher_alternative='greater',BH_test_count=len(a),finite_log_pq=True)
    if name.startswith('fold_'):
      fold=int(name.split('_')[-1])
      old=readcsv(A/f'09_temporal_remining_full_experiment/02_fold_rules/fold_{fold}/secondary_rules_strict_filtered.csv')
      oldset=set(old.antecedent);newset=set(strict.antecedent)
      summary['original_strict_rules']=len(old);summary['exact_strict_set_match']=oldset==newset
      summary['added_vs_old']=len(newset-oldset);summary['removed_vs_old']=len(oldset-newset)
      both=strict.merge(old[['antecedent','antecedent_support_count','positive_support_count']],on='antecedent',suffixes=('_new','_old'))
      assert (both.antecedent_support_count_new==both.antecedent_support_count_old).all()
      assert (both.positive_support_count_new==both.positive_support_count_old).all()
      if oldset!=newset:
        save_json(out/'set_difference.json',dict(added=sorted(newset-oldset),removed=sorted(oldset-newset)))
    save_json(done,summary);log('mining',json.dumps(summary,ensure_ascii=False));return strict

