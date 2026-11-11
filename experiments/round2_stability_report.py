"""Summarize actual supplemental runs and draw Times New Roman figures."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript_revision/round2_20261026/analysis/stability'
FONT=Path('C:/Windows/Fonts/times.ttf')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'mathtext.fontset':'stix',
                     'axes.linewidth':.7,'lines.linewidth':1.4,'pdf.fonttype':42,'ps.fonttype':42,
                     'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,default=str),encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def casedir(a):
    suffix='_fullfine' if a.get('full_fine',False) else ''
    return OUT/f"f{a['feeder']}_h{a['hour']}_{a['event']}_tau{a['factor']:g}_dt{a['step']:g}_te{a['event_time']:g}{suffix}"


def path_for(feeder,hour,event,factor=1,step=.005,full=False):
    return casedir(dict(feeder=feeder,hour=hour,event=event,factor=factor,step=step,event_time=180,full_fine=full))


def main():
    matrix=read(OUT/'dynamic_matrix.json')
    status=read(OUT/'dynamic_batch_status.json')
    dyn=[]; expected=[]
    for a in matrix:
        folder=casedir(a); p=folder/'metrics.json'
        expected.append(dict(arguments=a,present=p.exists()))
        if not p.exists(): continue
        m=read(p)
        row={k:v for k,v in m.items() if not isinstance(v,list)}
        row.update(feeder=a['feeder'],hour=a['hour'],full_fine=a['full_fine'],folder=str(folder.relative_to(ROOT)))
        lin=np.load(folder/'linearization.npz'); names=read(folder/'state_names.json')
        A=lin['state_matrix']; unused=[i for i,n in enumerate(names) if 'PIQ_xi ' in n or 'PIV_xi ' in n]
        active=[i for i in range(len(names)) if i not in unused]
        coupling=float(np.max(np.abs(A[np.ix_(active,unused)])))
        mu=np.linalg.eigvals(A[np.ix_(active,active)])
        row.update(disconnected_pi_states=len(unused),pi_to_active_coupling_max=coupling,
                   active_subsystem_spectral_abscissa=float(mu.real.max()))
        assert coupling<1e-10
        dyn.append(row)
    d=pd.DataFrame(dyn)
    d.to_csv(OUT/'dynamic_summary.csv',index=False)
    static=[]
    for p in sorted((OUT/'static').glob('*/metrics.json')):
        static.append(read(p))
    sf=pd.DataFrame(static);sf.to_csv(OUT/'static_summary.csv',index=False)
    checks=[]
    for feeder in [33,141]:
        for event in ['q_replay','load_step']:
            p=path_for(feeder,12,event);fine=path_for(feeder,12,event,step=.0025)
            if not (p/'trajectory.npz').exists() or not (fine/'trajectory.npz').exists(): continue
            checks.append(compare(p,fine,'half_post_event_step'))
    coarse=path_for(33,12,'q_replay'); fine=path_for(33,12,'q_replay',full=True)
    if (fine/'trajectory.npz').exists():checks.append(compare(coarse,fine,'fine_unforced_hold'))
    pd.DataFrame(checks).to_csv(OUT/'integration_checks.csv',index=False)
    cpf=[]
    for f in [33,141]:
        a=sf[(sf.feeder==f)&(sf.hour==12)&(sf.dispatch=='recorded_policy')]
        if len(a)==2:
            vals=a.fold_load_factor_estimate.to_numpy()
            cpf.append(dict(feeder=f,absolute_fold_multiplier_difference=float(np.ptp(vals)),
                            relative_fold_multiplier_difference=float(np.ptp(vals)/vals.mean())))
    pd.DataFrame(cpf).to_csv(OUT/'continuation_checks.csv',index=False)
    nominal=d[(d.tau_factor==1)&(d.integration_step_s==.005)&(~d.full_fine)] if len(d) else d
    if len(nominal)==12: plot_dynamic()
    if len(sf)==14: plot_static()
    manifest=dict(expected_dynamic_runs=len(matrix),recorded_dynamic_runs=len(d),
                  expected_static_runs=14,recorded_static_runs=len(sf),
                  failures=[x for x in status if 'failure' in x],
                  integration_checks=checks,continuation_checks=cpf,
                  source_hashes={},font='Times New Roman',font_file=str(FONT),
                  parameter_provenance='protocol/stability_scope_and_model.md',
                  claim_scope='WECC benchmark held-command interface; not closed-loop RNN stability or EMT')
    for p in [ROOT/'experiments/round2_stability_dynamic.py',ROOT/'experiments/round2_stability_static.py',
              ROOT/'experiments/round2_stability_report.py',
              ROOT/'manuscript_revision/round2_20261026/protocol/stability_scope_and_model.md',
              OUT/'runtime/andes/models/renewable/regca1.py',OUT/'runtime/andes/models/renewable/reeca1.py']:
        manifest['source_hashes'][str(p.relative_to(ROOT))]=sha(p)
    write(OUT/'analysis_manifest.json',manifest)
    if len(nominal):
        keys=['feeder','hour','event','voltage_min_pu','q_settling_s','voltage_settling_s',
              'endpoint_voltage_error_max_pu','active_subsystem_spectral_abscissa']
        print(nominal[keys].to_string(index=False))
    print('dynamic',len(d),'static',len(sf),'integration',len(checks),'failures',len(manifest['failures']))


def compare(coarse,fine,label):
    a=np.load(coarse/'trajectory.npz');b=np.load(fine/'trajectory.npz')
    prov=read(coarse/'provenance.json');t=a['time_s'];mask=t>=179.
    dv=np.column_stack([np.interp(t[mask],b['time_s'],b['voltage_pu'][:,j]) for j in range(b['voltage_pu'].shape[1])])-a['voltage_pu'][mask]
    dq=np.column_stack([np.interp(t[mask],b['time_s'],b['q_mvar'][:,j]) for j in range(b['q_mvar'].shape[1])])-a['q_mvar'][mask]
    ev=float(np.max(np.abs(dv)));eq=float(np.max(np.abs(dq)/np.asarray(prov['s_max_mva'])))
    return dict(case=coarse.name,comparison=label,max_voltage_difference_pu=ev,
                max_q_difference_rated_pu=eq,passed=ev<1e-5 and eq<1e-4)


def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png',bbox_inches='tight')
    plt.close(fig)


def plot_dynamic():
    colors=['#1c4f87','#a33e2d','#246a48'];hours=[12,18,21]
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.4),constrained_layout=True)
    for col,f in enumerate([33,141]):
        for h,c in zip(hours,colors):
            folder=path_for(f,h,'load_step');z=np.load(folder/'trajectory.npz')
            t=z['time_s']-180;sel=(t>=-.1)&(t<=1.5)
            axs[0,col].plot(t[sel],z['voltage_pu'][sel].min(1),color=c,label=f'{h:02}:00')
            folder=path_for(f,h,'q_replay');z=np.load(folder/'trajectory.npz');m=read(folder/'metrics.json');p=read(folder/'provenance.json')
            t=z['time_s']-180;sel=(t>=-.1)&(t<=1.5)
            err=100*np.max(np.abs(z['q_mvar']-np.asarray(m['final_q_target_mvar']))/np.asarray(p['s_max_mva']),axis=1)
            axs[1,col].plot(t[sel],err[sel],color=c,label=f'{h:02}:00')
        axs[0,col].axhline(.95,color='#555555',ls=':',lw=.9,label='0.95 p.u.')
        axs[0,col].set_title(f'({chr(97+col)}) {f}-bus: +10% load step')
        axs[1,col].set_title(f'({chr(99+col)}) {f}-bus: next held Q command')
        axs[0,col].set_ylabel('Minimum bus voltage (p.u.)')
        axs[1,col].set_ylabel('Maximum Q error (% of rating)')
        for ax in axs[:,col]:
            ax.set_xlabel('Time after the 180 s event (s)')
            ax.grid(alpha=.18,lw=.6);ax.legend(frameon=False,fontsize=8,loc='best')
    save(fig,'fig_stability_dynamic')


def plot_static():
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),constrained_layout=True)
    colors=['#1c4f87','#a33e2d','#246a48']
    for ax,f in zip(axs,[33,141]):
        for h,c in zip([12,18,21],colors):
            for dispatch,style in [('policy','-'),('zeroq','--')]:
                d=pd.read_csv(OUT/'static'/f'f{f}_h{h}_{dispatch}_ds0.04'/'continuation.csv')
                peak=d.additional_load_fraction.idxmax();d=d.iloc[:peak+1]
                ax.plot(1+d.additional_load_fraction,d.vmin_pu,color=c,ls=style,
                        label=f'{h:02}:00, '+('recorded Q' if dispatch=='policy' else 'zero Q'))
                ax.plot(1+d.additional_load_fraction.iloc[-1],d.vmin_pu.iloc[-1],'o',color=c,ms=3)
        ax.axhline(.95,color='#555555',ls=':',lw=.9)
        ax.set(title=f'{f}-bus feeder',xlabel='Uniform load multiplier',ylabel='Minimum bus voltage (p.u.)')
        ax.grid(alpha=.18,lw=.6);ax.legend(frameon=False,fontsize=7.5,loc='lower left')
    save(fig,'fig_static_loadability')


if __name__=='__main__':main()
