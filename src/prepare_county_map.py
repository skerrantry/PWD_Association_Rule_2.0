from common import *
import pyogrio
from pyogrio.raw import read

PROVINCES={'11','12','13','14','15','21','22','23','31','32','33','34','35','36','37','41','42','43','44','45','46','50','51','52','53','54','61','62','63','64','65'}

def run():
    complete=C/'county_map_manifest.json'
    if complete.exists():log('county','Completed source-key county map reused');return
    keys=np.load(C/'source_key.npy',mmap_mode='r');order=np.load(C/'source_key_order.npy',mmap_mode='r')
    sortedkeys=np.asarray(keys[order]);county=np.lib.format.open_memmap(C/'county_id.npy',mode='w+',dtype='int32',shape=(N,));county[:]=-1
    groups={};labels=[];total=0
    gdb=F/'Sub_Pine_All.gdb'
    for layer,_ in pyogrio.list_layers(gdb):
      if not re.fullmatch(r'T\d+',layer):continue
      meta,fids,_,arrays=read(gdb,layer=layer,columns=['xian'],read_geometry=False,return_fids=True)
      x=pd.Series(arrays[0]).fillna('').astype(str).str.replace(r'\.0$','',regex=True)
      rawcodes,unique=pd.factorize(x,sort=True)
      local=[]
      for code in unique:
        pc=layer[1:] if layer[1:] in PROVINCES else (code[:2] if code[:2] in PROVINCES else f'{layer}_未定位')
        key=f'{pc}:{code if code else "缺失"}'
        if key not in groups:
          groups[key]=len(groups);labels.append(dict(county_group_id=groups[key],province_group=pc,
              county_code_raw=code,county_key=key,administrative_code_verified=False))
        local.append(groups[key])
      ids=np.asarray(local,dtype=np.int32)[rawcodes]
      sourcekeys=(np.uint64(int(layer[1:]))<<np.uint64(32))|fids.astype(np.uint64)
      pos=np.searchsorted(sortedkeys,sourcekeys)
      assert (pos<N).all() and (sortedkeys[pos]==sourcekeys).all(),f'GDB/CSV key mismatch: {layer}'
      rows=np.asarray(order[pos]);assert (county[rows]==-1).all()
      county[rows]=ids;total+=len(ids)
      log('county',f'{layer}: mapped {len(ids):,}; cumulative {total:,}')
    assert total==N and (county>=0).all()
    county.flush();sh=np.nan_to_num(np.load(C/'SHOUFA_YEA.npy',mmap_mode='r'),nan=0).round().astype(int)
    sizes=np.bincount(county,minlength=len(groups));positives=np.bincount(county[(sh>=2001)&(sh<=2020)],minlength=len(groups))
    frame=pd.DataFrame(labels);frame['source_records']=sizes;frame['first_records_2001_2020']=positives
    frame.to_csv(C/'county_groups.csv',index=False,encoding='utf-8-sig')
    writecsv(frame,'19_原始县级字段分组及样本数量.csv')
    save_json(complete,dict(source_rows=total,groups=len(groups),source_key_join_complete=True,
      grouping='original xian field within resolved province group; codes are not assumed historically harmonized administrative boundaries'))
    log('county',f'Complete: {total:,} unique source-key matches / {len(groups)} raw county-field groups')

if __name__=='__main__':run()
