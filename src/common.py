from env import *
import hashlib,math,re
C=OLD;C.mkdir(parents=True,exist_ok=True)
A=ROOT/'inputs'
RAW=resolve('raw_csv')
N=int(CFG['source_units'])
YEARS=list(range(2001,2021));FOLDS={i+1:list(range(2001+i,2021,5)) for i in range(5)}
WINDOWS={1:(list(range(2001,2006)),list(range(2006,2011))),2:(list(range(2001,2011)),list(range(2011,2016))),3:(list(range(2001,2016)),list(range(2016,2021)))}
SEED=20261109
def save_json(path, obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def log(step, message):
    line=f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    print(line,flush=True)
    with (W/'logs'/f'{step}.log').open('a',encoding='utf-8') as f:f.write(line+'\n')

def readcsv(path, **kw):return pd.read_csv(path,encoding='utf-8-sig',**kw)
def writecsv(df, name):
    path=O/'数据表'/name
    df.to_csv(path,index=False,encoding='utf-8-sig')
    return path
def family_key(items):
    return ';'.join(sorted(re.sub(r'_(20\d{2})_', '_', x) for x in items))
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def metrics(n,a,total,positive):
    confidence=a/n if n else float('nan'); base=positive/total if total else float('nan')
    lift=confidence/base if base>0 else float('nan')
    b=n-a;c=positive-a;d=total-n-c
    return dict(antecedent_support_count=int(n),positive_support_count=int(a),
      confidence=confidence,base_rate=base,lift=lift,
      odds_ratio=((a+.5)*(d+.5))/((b+.5)*(c+.5)),
      wracc=(a/total-n/total*base) if total else float('nan'),
      risk_total=int(total),risk_positive_total=int(positive))
def packed_int(mask):return int.from_bytes(np.packbits(mask,bitorder='little').tobytes(),'little')
def unpack_int(value,n=N):
    return np.unpackbits(np.frombuffer(value.to_bytes((n+7)//8,'little'),dtype=np.uint8),bitorder='little')[:n].astype(bool)
def source_metadata(path):
    p=Path(path);st=p.stat();return dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns)

