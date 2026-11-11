"""Training-only weak-bus diagnosis at inherited PV locations, no siting search."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pandapower as pp
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
HOURS=[0,3,6,9,12,15,18,21]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def savejson(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def style():
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
        'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'stix','axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'legend.frameon':False,'xtick.labelsize':8,'ytick.labelsize':8,'savefig.dpi':300})

def sampled_source(path,days):
    chunks=[];first=None;maxima=None
    for frame in pd.read_csv(path,chunksize=50000):
        dates=pd.to_datetime(frame.iloc[:,0]);first=dates.iloc[0].normalize() if first is None else first
        delta=(dates-first).dt.days
        prefix=frame.loc[delta<999].iloc[:,1:].to_numpy(float)
        if len(prefix):maxima=prefix.max(0) if maxima is None else np.maximum(maxima,prefix.max(0))
        mask=delta.isin(days)&dates.dt.hour.isin(HOURS)&(dates.dt.minute==0)
        chosen=frame.loc[mask].copy();chosen.iloc[:,0]=dates.loc[mask].astype(str)
        chunks.append(chosen)
        if delta.iloc[-1]>=999:break
    out=pd.concat(chunks,ignore_index=True)
    return out.iloc[:,0].to_numpy(str),out.iloc[:,1:].to_numpy(float),maxima

def tree_layout(graph,root):
    pos={};cursor=[0]
    def place(node,parent,depth):
        children=sorted(n for n in graph[node] if n!=parent)
        ys=[place(c,node,depth+1) for c in children]
        if not ys: y=cursor[0];cursor[0]+=1
        else:y=float(np.mean(ys))
        pos[node]=(depth,y);return y
    if not nx.is_tree(graph):raise ValueError('Expected a connected radial benchmark')
    place(root,None,0);return pos

def render(out,scenario,net,table,v,ts,sens):
    style();nb=len(net.bus);pv=set(net.sgen.bus.astype(int));slack=int(net.ext_grid.bus.iloc[0])
    graph=nx.Graph();graph.add_nodes_from(net.bus.index.astype(int))
    for r in net.line.itertuples():
        if r.in_service:graph.add_edge(int(r.from_bus),int(r.to_bus))
    pos=tree_layout(graph,slack);top=table.loc[table.bus_index!=slack].head(12)
    fig,axes=plt.subplots(1,2,figsize=(7.18,3.6),gridspec_kw={'width_ratios':[1.5,1]},layout='constrained')
    ax=axes[0];nx.draw_networkx_edges(graph,pos,ax=ax,width=.6,edge_color='#888888')
    nodelist=sorted(int(b) for b in graph.nodes)
    risk_by_bus=table.set_index('bus_index').violation_pct.to_dict()
    node_colors=np.asarray([risk_by_bus[b] for b in nodelist])
    expected=100*((v<.95)|(v>1.05)).mean(0)
    for b,c in zip(nodelist,node_colors):
        assert np.isclose(c,expected[list(net.bus.index).index(b)],atol=1e-12)
    pd.DataFrame({'nodelist_position':range(len(nodelist)),'bus_index':nodelist,
        'plotted_violation_pct':node_colors}).to_csv(out/'figure_node_color_audit.csv',index=False)
    colors=nx.draw_networkx_nodes(graph,pos,nodelist=nodelist,ax=ax,node_size=22 if nb>40 else 48,
        node_color=node_colors,cmap='YlOrRd',vmin=0,vmax=100,linewidths=.3,edgecolors='#555555')
    nx.draw_networkx_nodes(graph,pos,nodelist=sorted(pv),ax=ax,node_size=55 if nb>40 else 90,node_shape='s',
        node_color='none',edgecolors='#126052',linewidths=1.)
    # Off-node labels leave risk colors visible; split crowded branch tips into two rows.
    labelled=sorted([int(b) for b in top.bus_index.head(6)],key=lambda b:(pos[b][1],pos[b][0]))
    for group,sign in [(labelled[:3],-1),(labelled[3:],1)]:
        group=sorted(group,key=lambda b:pos[b][0]);center=float(np.mean([pos[b][0] for b in group]))
        row=float(np.mean([pos[b][1] for b in group]))+sign*(2.3 if nb>40 else .8)
        spacing=2.7 if nb>40 else 1.6
        for j,b in enumerate(group):
            destination=(center+(j-(len(group)-1)/2)*spacing,row)
            ax.annotate(str(b),xy=pos[b],xytext=destination,ha='center',va='center',fontsize=7,
                arrowprops=dict(arrowstyle='-',color='#333333',lw=.6,shrinkA=1,shrinkB=4))
    ax.set_axis_off();ax.set_title('(a) Fixed PV locations and voltage-risk pattern')
    ax.plot([],[],marker='s',markerfacecolor='none',markeredgecolor='#126052',ls='none',label='Inherited PV bus')
    ax.legend(loc='lower left',fontsize=7)
    cb=fig.colorbar(colors,ax=ax,location='bottom',fraction=.06,pad=.02);cb.set_label('Bus voltage violation rate [%]')
    ax=axes[1];y=np.arange(len(top))
    ax.barh(y,top.undervoltage_pct,color='#2C6095',label='Undervoltage')
    ax.barh(y,top.overvoltage_pct,left=top.undervoltage_pct,color='#B64B45',label='Overvoltage')
    ax.set_yticks(y,[str(b)+(' *' if b in pv else '') for b in top.bus_index]);ax.invert_yaxis()
    ax.set_xlim(0,100);ax.set_xlabel('Violation rate [%]');ax.set_ylabel('Benchmark bus index')
    ax.set_title('(b) Highest-risk buses');ax.legend(fontsize=7,loc='lower right')
    ax.text(.02,-.18,'* PV-connected bus; indices are zero-based',transform=ax.transAxes,fontsize=7)
    fig.savefig(out/'weak_bus_ranking.pdf',bbox_inches='tight');fig.savefig(out/'weak_bus_ranking.png',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.18,2.8),layout='constrained')
    times=pd.to_datetime(ts);hour_risk=np.array([np.nanmean(((v[times.hour==h]<.95)|(v[times.hour==h]>1.05)).astype(float),axis=0)*100 for h in HOURS])
    im=ax.imshow(hour_risk,aspect='auto',origin='lower',interpolation='nearest',extent=[-.5,nb-.5,-.5,7.5],vmin=0,vmax=100,cmap='YlOrRd')
    ax.set_yticks(range(8),[f'{h:02d}:00' for h in HOURS]);ax.set_xlabel('Benchmark bus index');ax.set_ylabel('Source time')
    fig.colorbar(im,ax=ax,label='Training-snapshot violation rate [%]');fig.savefig(out/'weak_bus_time_heatmap.pdf');fig.savefig(out/'weak_bus_time_heatmap.png');plt.close(fig)

def run(scenario,output,days):
    out=output/scenario;out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'environments/var_voltage_control/data'/scenario
    spec=dict(analysis='Training-only diagnosis at inherited benchmark PV positions',scenario=scenario,
        pv_scale=1.5,days=days,hours=HOURS,samples_per_day=8,weights='equal weight per complete snapshot',control='zero inverter reactive output',
        voltage_bounds_pu=[.95,1.05],rank='descending violation frequency, then mean violation magnitude, then ascending benchmark bus index',
        scope='Empirical steady-state voltage weakness, not a dynamic stability index or placement optimization',
        date_partition='days 1..998 within training prefix; no validation or test points used',
        sensitivity='central finite difference in each existing inverter Q, 0.1% of instantaneous headroom capped at 0.001 Mvar; check against half-step',
        runtime=dict(python=sys.version,pandapower=pp.__version__,numpy=np.__version__,pandas=pd.__version__),
        source_files={n:dict(path=str(data/n),size=(data/n).stat().st_size,sha256=sha(data/n)) for n in ['model.p','pv_active.csv','load_active.csv','load_reactive.csv']})
    savejson(out/'protocol.json',spec)
    cache=out/'source_snapshots.npz'
    if cache.exists():
        with np.load(cache) as z:ts=z['timestamps'];p=z['pv_p'];lp=z['load_p'];lq=z['load_q'];ratings=z['s_max']
    else:
        print(f'{scenario}: reading training-only profiles',flush=True)
        ts,p,peak=sampled_source(data/'pv_active.csv',days);t2,lp,_=sampled_source(data/'load_active.csv',days);t3,lq,_=sampled_source(data/'load_reactive.csv',days)
        if not np.array_equal(ts,t2) or not np.array_equal(ts,t3):raise ValueError('profile timestamps disagree')
        p*=1.5;ratings=1.2*1.5*peak
        np.savez_compressed(cache,timestamps=ts,pv_p=p,load_p=lp,load_q=lq,s_max=ratings)
    if len(ts)!=len(days)*8:raise ValueError(f'Incomplete training snapshot set: {len(ts)}')
    net=pp.from_pickle(data/'model.p');nb=len(net.bus)
    checkpoint=out/'voltage_scan_progress.npz'
    v=np.full((len(ts),nb),np.nan);done=np.zeros(len(ts),bool)
    if checkpoint.exists():
        with np.load(checkpoint) as z:v=z['voltage'];done=z['attempted']
    failures=[];started=time.perf_counter()
    for k in np.flatnonzero(~done):
        net.sgen['p_mw']=p[k];net.sgen['q_mvar']=0.;net.load['p_mw']=lp[k];net.load['q_mvar']=lq[k]
        try:
            pp.runpp(net,numba=True,init='flat',tolerance_mva=1e-9,max_iteration=50)
            v[k]=net.res_bus.vm_pu.sort_index().to_numpy()
        except pp.LoadflowNotConverged as e:failures.append(dict(index=int(k),timestamp=str(ts[k]),error=str(e)))
        done[k]=True
        if (k+1)%500==0 or k==len(ts)-1:
            np.savez_compressed(checkpoint,voltage=v,attempted=done)
            print(f'{scenario}: {k+1}/{len(ts)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    if not np.isfinite(v).all():raise RuntimeError('Unsolved snapshots retained in checkpoint; do not treat as compliant')
    under=v<.95;over=v>1.05;mag=np.maximum(.95-v,0)+np.maximum(v-1.05,0)
    pv=set(net.sgen.bus.astype(int));slack=int(net.ext_grid.bus.iloc[0])
    table=pd.DataFrame(dict(bus_index=net.bus.index,bus_name=net.bus.name.astype(str),zone=net.bus.zone.astype(str),
        pv_present=[int(i) in pv for i in net.bus.index],is_slack=[int(i)==slack for i in net.bus.index],
        undervoltage_pct=100*under.mean(0),overvoltage_pct=100*over.mean(0),violation_pct=100*(under|over).mean(0),
        mean_violation_pu=mag.mean(0),max_undervoltage_pu=np.maximum(.95-v,0).max(0),max_overvoltage_pu=np.maximum(v-1.05,0).max(0),
        minimum_voltage_pu=v.min(0),maximum_voltage_pu=v.max(0)))
    table=table.sort_values(['violation_pct','mean_violation_pu','bus_index'],ascending=[False,False,True]);table.insert(0,'risk_rank',range(1,nb+1));table.to_csv(out/'bus_risk_ranking.csv',index=False)
    equipment=pd.DataFrame(dict(inverter_index=net.sgen.index,bus_index=net.sgen.bus,bus_name=[str(net.bus.loc[b,'name']) for b in net.sgen.bus],
        zone=net.sgen.name,s_max_mva=ratings,pv_scale=1.5,source=str(data/'model.p'),origin='Inherited serialized benchmark; not selected from this diagnosis'))
    equipment.to_csv(out/'pv_location_provenance.csv',index=False)
    sens=[]
    for condition,k in [('minimum_voltage',int(np.argmin(v.min(1)))),('maximum_voltage',int(np.argmax(v.max(1))))]:
        net.sgen['p_mw']=p[k];net.load['p_mw']=lp[k];net.load['q_mvar']=lq[k];net.sgen['q_mvar']=0.
        cap=np.sqrt(np.maximum(ratings**2-p[k]**2,0))
        for j in range(len(net.sgen)):
            delta=min(.001,.001*cap[j])
            if delta<1e-8:continue
            slopes=[]
            for d in [delta,delta/2]:
                vv=[]
                for sign in [-1,1]:
                    net.sgen['q_mvar']=0.;net.sgen.loc[net.sgen.index[j],'q_mvar']=sign*d
                    pp.runpp(net,numba=True,init='flat',tolerance_mva=1e-10,max_iteration=50);vv.append(net.res_bus.vm_pu.sort_index().to_numpy(copy=True))
                slopes.append((vv[1]-vv[0])/(2*d))
            for ib,bus in enumerate(net.bus.index):sens.append(dict(condition=condition,timestamp=ts[k],snapshot_index=k,
                inverter_index=j,inverter_bus=int(net.sgen.bus.iloc[j]),bus_index=int(bus),delta_q_mvar=delta,
                dv_dq_pu_per_mvar=float(slopes[0][ib]),half_step_slope=float(slopes[1][ib]),absolute_step_check=float(abs(slopes[0][ib]-slopes[1][ib]))))
    pd.DataFrame(sens).to_csv(out/'voltage_q_sensitivity.csv',index=False)
    graph=nx.Graph();graph.add_nodes_from(net.bus.index.astype(int))
    for r in net.line.itertuples():
        if r.in_service:graph.add_edge(int(r.from_bus),int(r.to_bus))
    table['nearest_pv_hops']=[min(nx.shortest_path_length(graph,int(b),q) for q in pv) for b in table.bus_index]
    table.to_csv(out/'bus_risk_ranking.csv',index=False)
    summary=dict(scenario=scenario,snapshots=len(ts),failed_snapshots=int(np.sum(~np.isfinite(v).all(1))),
        empirical_all_bus_vrate_pct=float(100*(under|over).mean()),under_pct=float(100*under.mean()),over_pct=float(100*over.mean()),
        highest_risk_buses=table.loc[~table.is_slack].head(10).bus_index.astype(int).tolist(),
        highest_risk_pv_overlap=int(table.loc[~table.is_slack].head(10).pv_present.sum()),
        sensitivity_max_half_step_difference=max(r['absolute_step_check'] for r in sens),
        figure_bus_indices='zero-based original benchmark index; bus_name retained separately in CSV',
        conclusion_scope='Training-grid empirical risk and local static Q sensitivity at fixed existing PV sites only')
    savejson(out/'summary.json',summary);render(out,scenario,net,table,v,ts,sens)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--scenario',choices=['case33_3min_final','case141_3min_final','both'],default='both')
    ap.add_argument('--output',type=Path,default=ROOT/'manuscript_revision/round2_20261026/analysis/weak_bus')
    ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    for scenario in (['case33_3min_final','case141_3min_final'] if a.scenario=='both' else [a.scenario]):run(scenario,a.output.resolve(),[1,500,998] if a.smoke else list(range(1,999)))
