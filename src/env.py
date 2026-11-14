from pathlib import Path
import sys,json,time,csv,os
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
config_path=Path(os.environ.get('PWD_CONFIG',str(ROOT/'config.json'))).resolve()
if not config_path.exists():raise FileNotFoundError('Copy config.example.json to config.json and set private input paths, or set PWD_CONFIG.')
CFG=json.loads(config_path.read_text(encoding='utf-8'))
def resolve(k):
    p=Path(CFG[k]);return p if p.is_absolute() else config_path.parent/p
W=resolve('work_dir');O=resolve('output_dir');F=resolve('raw_backup_root');OLD=resolve('numeric_cache')
for p in [W/'cache',W/'logs',O/'核查与数据表',O/'附属材料',O/'图',O/'数据表']:p.mkdir(parents=True,exist_ok=True)
def log(s):
    line=f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {s}'
    print(line,flush=True)
    with (W/'logs/progress.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
def js(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def outcsv(df,name):df.to_csv(O/'核查与数据表'/name,index=False,encoding='utf-8-sig')
