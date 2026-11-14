from pathlib import Path
import subprocess,sys,os,argparse
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--config',default=str(root/'config.json'));p.add_argument('--stage',choices=['cache','repair','analysis','validation','uncertainty','figures'],required=True);a=p.parse_args()
os.environ['PWD_CONFIG']=str(Path(a.config).resolve())
steps={'cache':[('prepare_numeric_cache.py',[]),('prepare_county_map.py',[])],
'repair':[('audit_upstream.py',[]),('repair_labels.py',[]),('refine_spatial.py',[]),('fix_prebaseline.py',[]),('build_spatial_index.py',[]),('final_linkage_qa.py',[]),('fix_reference_coordinates.py',[]),('geometry_measurement_audit.py',[]),('boundary_full.py',[]),('rebuild_history_features.py',['--pressure-only'])],
'analysis':[('new_analysis.py',['--dataset-and-main'])],'validation':[('validate_new.py',[])],'uncertainty':[('new_uncertainty.py',[])],'figures':[('plot_new_results.py',[])]}
for name,args in steps[a.stage]:subprocess.run([sys.executable,str(root/'src'/name),*args],check=True,cwd=root)
