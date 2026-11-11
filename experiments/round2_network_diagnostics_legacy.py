"""Replay preserved historical checkpoints without changing the historical code.

Run in a fresh Python process. The legacy repository is intentionally first on
sys.path. Outputs are newly reconstructed trajectories, not original records.
"""
from __future__ import annotations
import argparse, contextlib, copy, hashlib, io, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY = ROOT.parents[1]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--legacy-root',type=Path,default=DEFAULT_LEGACY)
    ap.add_argument('--output',type=Path,default=ROOT/'manuscript_revision/round2_20261026/analysis/legacy_residual')
    ap.add_argument('--days',type=int,default=30)
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    a=ap.parse_args(); a.output=a.output.resolve(); a.output.mkdir(parents=True,exist_ok=True)
    legacy=a.legacy_root.resolve(); sys.path.insert(0,str(legacy)); os.chdir(legacy)
    import numpy as np
    import pandas as pd
    import torch as th
    import pandapower as pp
    th.set_num_threads(1)
    from experiments.run_q1_ml_compare import load_env_config,load_alg_config,make_args,make_env
    from models.model_registry import Model
    from utilities.util import prep_obs,translate_action
    scenario='case141_3min_final'
    oldroot=legacy/'exp/q1_multiseed_1000ep_case141'/scenario/'matd3'
    config=load_env_config(scenario,1.5,mode='distributed')
    config.update(seed=0,voltage_weight=1.,violation_weight=0.,q_weight=.1,start_day_candidates=list(range(1030,1060)))
    print('Loading historical environment',flush=True)
    np.random.seed(0);th.manual_seed(0)
    env=make_env(config)
    alg=load_alg_config('matd3',config)
    args=make_args(config,alg,env,dict(max_steps=240,cuda=False))
    cls=Model['matd3']; target=cls(args) if args.target else None
    model=cls(args,target) if target is not None else cls(args)
    model.eval()
    protocol=dict(kind='new replay of preserved original checkpoints',legacy_root=str(legacy),
        original_rng_state_available=False,reset_rng='reset NumPy seed to training seed before each replay; historical post-training reset RNG state unavailable',
        dates=list(range(1030,1030+a.days)),max_steps=240,start_hour=6,environment_config=config,
        python=sys.version,numpy=np.__version__,torch=th.__version__,pandapower=pp.__version__,
        source_sha256={str(p.relative_to(legacy)):sha(p) for p in [legacy/'experiments/run_q1_ml_compare.py',legacy/'experiments/run_risk_aware_multiseed.py',legacy/'environments/var_voltage_control/voltage_control_env.py',legacy/'models/matd3.py',legacy/'utilities/util.py',legacy/'args/default.yaml',legacy/'args/alg_args/matd3.yaml']},
        timestamp_note='Physical-input source timestamp is captured using historical row-index semantics. Manual reset starts at index 1, the first source sample is repeated, and termination occurs at steps=240 after 239 actions. Neither cadence nor number of points is repaired in this historical replay.')
    (a.output/'replay_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    all_summaries=[]
    for seed in a.seeds:
        out=a.output/f'seed_{seed}';out.mkdir(exist_ok=True)
        checkpoint=oldroot/f'seed_{seed}'/'best_model.pt'
        ck=th.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model_state_dict'],strict=True);model.eval()
        np.random.seed(seed);th.manual_seed(seed)
        saved=json.loads((oldroot/f'seed_{seed}'/'seed_result.json').read_text())
        rows=[];volts=[];ppv=[];qpv=[];loads=[];qloads=[];actions=[]
        worst=None;worst_value=2.
        started=time.perf_counter()
        for day in protocol['dates']:
            with contextlib.redirect_stdout(io.StringIO()): state,_=env.manual_reset(day,6,0)
            hidden=model.policy_dicts[0].init_hidden()
            for step in range(240):
                p=env.powergrid.sgen.p_mw.to_numpy(copy=True)
                lp=env.powergrid.load.p_mw.to_numpy(copy=True)
                lq=env.powergrid.load.q_mvar.to_numpy(copy=True)
                with th.inference_mode():
                    tensor=prep_obs(state).contiguous().view(1,args.agent_num,args.obs_size)
                    act,_,_,_,hidden=model.get_actions(tensor,status='test',exploration=False,actions_avail=th.tensor(env.get_avail_actions()),target=False,last_hid=hidden)
                    _,actual=translate_action(args,act,env)
                with contextlib.redirect_stdout(io.StringIO()): reward,done,info=env.step(actual,add_noise=False)
                v=env.powergrid.res_bus.vm_pu.sort_index().to_numpy(copy=True)
                q=env.powergrid.sgen.q_mvar.to_numpy(copy=True)
                capacity=np.sqrt(np.maximum(env.s_max**2-p**2,0))
                timestamp=env.pv_data.index[day*480+120+max(1,step)]
                row=dict(seed=seed,day=day,step=step,timestamp=str(timestamp),hour=timestamp.hour+timestamp.minute/60,
                    vrate=float(info['percentage_of_v_out_of_control']),uvrate=float(np.mean(v<.95)),ovrate=float(np.mean(v>1.05)),
                    vmin=float(v.min()),vmax=float(v.max()),total_pv_mw=float(p.sum()),total_load_mw=float(lp.sum()),
                    total_load_mvar=float(lq.sum()),total_q_mvar=float(q.sum()),total_headroom_mvar=float(capacity.sum()),
                    q_abs_utilization=float(np.abs(q).sum()/capacity.sum()),destroy=float(info['destroy']),reward=float(reward))
                rows.append(row);volts.append(v);ppv.append(p);qpv.append(q);loads.append(lp);qloads.append(lq);actions.append(np.asarray(actual))
                if v.min()<worst_value:
                    worst_value=float(v.min());worst=dict(row=row,p_mw=p.tolist(),q_mvar=q.tolist(),load_p_mw=lp.tolist(),load_q_mvar=lq.tolist(),s_max_mva=env.s_max.tolist(),voltage_pu=v.tolist(),alpha=float(config['action_scale']))
                state=env.get_obs()
                if done: break
            print(f'legacy seed={seed} day={day} rows={len(rows)} seconds={time.perf_counter()-started:.1f}',flush=True)
            pd.DataFrame(rows[-240:]).to_csv(out/'last_day_progress.csv',index=False)
        frame=pd.DataFrame(rows);frame.to_csv(out/'step_metrics.csv',index=False)
        np.savez_compressed(out/'trajectory.npz',voltage=np.asarray(volts),pv_p=np.asarray(ppv),pv_q=np.asarray(qpv),load_p=np.asarray(loads),load_q=np.asarray(qloads),normalized_scaled_action=np.asarray(actions),s_max=env.s_max,bus_index=env.powergrid.bus.index.to_numpy(),pv_bus=env.powergrid.sgen.bus.to_numpy())
        (out/'worst_undervoltage_snapshot.json').write_text(json.dumps(worst,indent=2),encoding='utf8')
        summary=dict(seed=seed,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),selected_episode=int(ck['episode']),
            archived_vrate=saved['test_metrics']['percentage_of_v_out_of_control'],replayed_vrate=float(frame.vrate.mean()),
            replayed_uvrate=float(frame.uvrate.mean()),replayed_ovrate=float(frame.ovrate.mean()),points=len(frame),days=a.days,
            max_day_points=int(frame.groupby('day').size().max()),runtime_seconds=time.perf_counter()-started,
            vmin=float(frame.vmin.min()),vmax=float(frame.vmax.max()),destroy_count=int(frame.destroy.sum()),
            exact_original_statistic_reproduced=bool(a.days==30 and abs(float(frame.vrate.mean())-saved['test_metrics']['percentage_of_v_out_of_control'])<1e-10))
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');all_summaries.append(summary)
    (a.output/'replay_summary.json').write_text(json.dumps(all_summaries,indent=2),encoding='utf8')
    print(json.dumps(all_summaries,indent=2),flush=True)

if __name__=='__main__':main()
