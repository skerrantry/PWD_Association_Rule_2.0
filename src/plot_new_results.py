from env import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import itertools,math
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.7})
FEATURES=['Bio8','Bio14','Bio18','LT_all','LT_inf','GDP','Dis_rd','W_rd_den_5','FO','DBH','Area','slope','inf_by','inf_press_3km']
LABELS=['Bio8','Bio14','Bio18','Timber inflow','Suspect-timber inflow','GDP','Road distance','Road density','Forest origin','DBH','Recorded area','Slope class','Shared-boundary history','Decayed historical pressure']
def variables(f):
    return {next((x for x in sorted(FEATURES,key=len,reverse=True) if token.startswith(x+'_')),token) for token in f.split(';')}
def save(fig,name):
    root=O/'图'/name
    fig.savefig(str(root)+'.png',dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(str(root)+'.svg',bbox_inches='tight',facecolor='white')
    fig.savefig(str(root)+'.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(str(root)+'.tiff',dpi=600,bbox_inches='tight',facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
def run():
    data=pd.read_csv(O/'核查与数据表/32_完整验证后的新规则族结果.csv')
    selected=data[data.temporal_class=='strong_internal_temporal'].sort_values('secondary_score',ascending=False)
    groups=[selected[selected.spatial_scope==s].copy() for s in ['broad_multi_province','spatially_concentrated']]
    names=['Broad provincial coverage','Spatially concentrated'];colors=['#345E78','#B27D55'];freq=[]
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.5),sharey=True,layout='constrained')
    for ax,g,name,color in zip(axes,groups,names,colors):
        sets=g.rule_family.map(variables);counts=np.array([sum(x in s for s in sets) for x in FEATURES]);values=100*counts/len(g)
        ax.barh(np.arange(14),values,color=color,height=.7);ax.set_yticks(np.arange(14),LABELS);ax.set_xlim(0,100)
        ax.set_xlabel('Families containing feature (%)');ax.set_title(f'{name}\nn = {len(g)} families',fontsize=9)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for y,v,c in zip(range(14),values,counts):ax.text(v+.7,y,str(c),va='center',fontsize=6.5)
        for f,c,v in zip(FEATURES,counts,values):freq.append(dict(scope=name,feature=f,number_of_families=int(c),denominator=len(g),percent=float(v)))
    axes[0].invert_yaxis()
    save(fig,'Figure_1_新版特征构成');outcsv(pd.DataFrame(freq),'37_图1特征频数及分母.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,5.3),sharey=True,layout='constrained');matrixrows=[]
    for ax,g,name in zip(axes,groups,names):
        bins=np.array_split(np.arange(len(g)),5);values=np.zeros((14,5));counts=np.zeros((14,5),int)
        sets=g.rule_family.map(variables).tolist()
        for j,ids in enumerate(bins):
            for i,f in enumerate(FEATURES):counts[i,j]=sum(f in sets[k] for k in ids);values[i,j]=100*counts[i,j]/len(ids)
        im=ax.imshow(values,vmin=0,vmax=100,cmap='Blues',aspect='auto');ax.set_title(name,fontsize=9)
        ax.set_yticks(range(14),LABELS);ax.set_xticks(range(5),['1–20','21–40','41–60','61–80','81–100'],rotation=35,ha='right');ax.set_xlabel('Rank percentile band')
        for i,f in enumerate(FEATURES):
            for j,ids in enumerate(bins):
                ax.text(j,i,f'{values[i,j]:.0f}\n({counts[i,j]})',ha='center',va='center',fontsize=5.7,color='white' if values[i,j]>55 else 'black')
                matrixrows.append(dict(scope=name,feature=f,band=j+1,band_denominator=len(ids),count=int(counts[i,j]),percent=float(values[i,j])))
    fig.colorbar(im,ax=axes,shrink=.55,label='Families in rank band (%)');save(fig,'Figure_2_新版排名分布');outcsv(pd.DataFrame(matrixrows),'38_图2排名带计数.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.4),layout='constrained');edges=[]
    angles=np.linspace(np.pi/2,np.pi/2+2*np.pi,14,endpoint=False);positions=np.column_stack([np.cos(angles),np.sin(angles)])
    for ax,g,name in zip(axes,groups,names):
        sets=g.rule_family.map(variables).tolist();counts=np.array([sum(f in s for s in sets) for f in FEATURES])
        for i,j in itertools.combinations(range(14),2):
            count=sum(FEATURES[i] in s and FEATURES[j] in s for s in sets);pct=count/len(g)
            edges.append(dict(scope=name,feature_1=FEATURES[i],feature_2=FEATURES[j],cooccurring_families=count,denominator=len(g)))
            if pct>=.01:ax.plot(positions[[i,j],0],positions[[i,j],1],color='#8DA3AE',alpha=.15+.5*pct,lw=.2+4*pct,zorder=1)
        ax.scatter(positions[:,0],positions[:,1],s=35+650*counts/len(g),color='#5C7D8B',edgecolor='white',linewidth=.5,zorder=2)
        for i,f in enumerate(FEATURES):
            xx,yy=positions[i]*1.28
            ha='center' if abs(xx)<.1 else ('right' if xx<0 else 'left')
            va='bottom' if yy>.5 else ('top' if yy<-.5 else 'center')
            ax.text(xx,yy,f.replace('inf_press_3km','Pressure').replace('W_rd_den_5','Road density').replace('inf_by','Boundary').replace('Dis_rd','Road dist.'),ha=ha,va=va,fontsize=6.6,bbox={'facecolor':'white','edgecolor':'none','pad':.25})
        ax.set_xlim(-1.9,1.9);ax.set_ylim(-1.6,1.62);ax.set_aspect('equal');ax.axis('off');ax.set_title(f'{name}\nn = {len(g)}',fontsize=9)
    save(fig,'Figure_3_新版共现网络');outcsv(pd.DataFrame(edges),'39_图3完整共现边表.csv')
    province=pd.read_csv(O/'核查与数据表/20_最终省年风险集与未截尾敏感性计数.csv',dtype={'province':str})
    province=province[province.scope=='main_linkage_censored'];counts=province.pivot(index='province',columns='year',values='positive_records').fillna(0)
    fig,ax=plt.subplots(figsize=(7.2,6.3),layout='constrained');masked=np.ma.masked_less_equal(counts.to_numpy(),0)
    im=ax.imshow(masked,cmap='Blues',norm=LogNorm(vmin=1,vmax=max(2,float(masked.max()))),aspect='auto')
    ax.set_yticks(range(len(counts)),[str(v).replace('未定位','unresolved') for v in counts.index]);ax.set_xticks(range(20),range(2001,2021),rotation=90);ax.set_xlabel('Year');ax.set_ylabel('Province group code');ax.set_title('Available first-recorded events by province and year',fontsize=10)
    fig.colorbar(im,ax=ax,shrink=.65,label='Recorded events (log scale)');save(fig,'Figure_S1_省年记录热图')
    js(W/'cache/figures_complete.json',dict(figures=4,backend='Python matplotlib',main_catalogue='590 primary internally temporally robust families',comparison_denominators=[len(g) for g in groups],source_data_tables=['37','38','39','20'],interpretation='frequencies and cooccurrences, not causal interactions; blank province-year cells mean zero available records, not proven absence'))
    log('Four revised scientific figures and their source tables exported')
if __name__=='__main__':run()
