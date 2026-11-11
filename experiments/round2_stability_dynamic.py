"""Reproducible, exploratory WECC positive-sequence interface validation.

The learned supervisor is NOT advanced at the integration time step. Recorded
180 s setpoints are used as held references. This is a benchmark dynamic model,
not an identified hardware model or an EMT model.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript_revision/round2_20261026/analysis/stability'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT / 'runtime'))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import pandapower as pp
import andes
from experiments.run_revision_pilot import environment_config
from environments.var_voltage_control.voltage_control_env import VoltageControl


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_case(feeder=33, hour=12, day=1030):
    scenario = f'case{feeder}_3min_final'
    job = 'ej00001' if feeder == 33 else 'ej00721'
    bundle = ROOT / f'exp/revision_access_41572/evaluation/continuation_v02_io_recovery/jobs/{job}/attempt0001/result'
    meta = json.loads((bundle / 'evaluation.json').read_text())
    assert meta['evaluation_job']['run_id'] == f'reference_B2__{scenario}__seed0'
    source = bundle / 'trajectory/fixed_test_step_metrics.csv'
    df = pd.read_csv(source)
    row = df[(df.day_index == day) & (df.step_index == hour * 20)].iloc[0]
    nxt = df[(df.day_index == day) & (df.step_index == hour * 20 + 1)].iloc[0]
    cfg = environment_config(scenario, seed=0, phase='validation', points=480, start_hour=0)
    with contextlib.redirect_stdout(io.StringIO()):
        env = VoltageControl(cfg)
        env.manual_reset(day, 0, 0)
        env.steps = hour * 20
        env._set_demand_and_pv(add_noise=False)
    env._loaded_step = env.steps
    net = env.powergrid
    q = np.asarray(json.loads(row.inverter_q_mvar))
    np.testing.assert_allclose(net.sgen.p_mw, json.loads(row.inverter_p_mw), atol=1e-10)
    np.testing.assert_allclose(env.s_max, json.loads(row.inverter_s_max_mva), atol=1e-10)
    net.sgen['q_mvar'] = q
    pp.runpp(net, calculate_voltage_angles=True, tolerance_mva=1e-10, max_iteration=50)
    provenance = dict(feeder=feeder, day=day, hour=hour, seed=0,
                      evaluation=str(bundle/'evaluation.json'), evaluation_sha256=sha(bundle/'evaluation.json'),
                      trajectory=str(source), trajectory_sha256=sha(source), timestamp=row.timestamp,
                      next_q_mvar=json.loads(nxt.inverter_q_mvar), next_p_mw=json.loads(nxt.inverter_p_mw),
                      s_max_mva=env.s_max.tolist(), alpha=cfg['action_scale'],
                      voltage_pu=net.res_bus.vm_pu.tolist(), q_mvar=q.tolist())
    env.round2_q_next=np.asarray(json.loads(nxt.inverter_q_mvar))
    return env, provenance


def convert(env, factor=1., event='q_step', event_time=180., horizon=185., step=.005):
    """Keep original AC network and constant-PQ loads; add published WECC blocks."""
    n = env.powergrid
    s = andes.System(default_config=True, no_output=True, no_undill=True,
                     config={'mva':float(n.sn_mva), 'numba':0},
                     pycode_path=str(OUT/'pycode'))
    s.PQ.config.pq2z = 0
    s.PQ.config.p2p = s.PQ.config.q2q = 1.
    s.PQ.config.p2z = s.PQ.config.q2z = 0.
    s.PFlow.config.tol = 1e-10
    s.PFlow.config.max_iter = 100
    s.PFlow.config.report = 0
    s.TDS.config.tol = 1e-8
    s.TDS.config.tstep = step
    s.TDS.config.tf = horizon
    s.TDS.config.no_tqdm = 1
    s.TDS.config.criteria = 0
    sb = float(n.sn_mva)
    for idx, b in n.bus.iterrows():
        s.add('Bus', idx=int(idx)+1, Vn=float(b.vn_kv),
              v0=float(n.res_bus.loc[idx,'vm_pu']), a0=float(n.res_bus.loc[idx,'va_degree'])*np.pi/180.)
    for idx, b in n.line.iterrows():
        if not b.in_service:
            continue
        vn = float(n.bus.loc[b.from_bus,'vn_kv'])
        zb = vn*vn/sb
        s.add('Line', idx=f'L{idx}', bus1=int(b.from_bus)+1, bus2=int(b.to_bus)+1,
              Sn=sb, Vn1=vn, Vn2=float(n.bus.loc[b.to_bus,'vn_kv']),
              r=float(b.r_ohm_per_km*b.length_km/b.parallel/zb),
              x=float(b.x_ohm_per_km*b.length_km/b.parallel/zb),
              b=float(2*np.pi*n.f_hz*b.c_nf_per_km*1e-9*b.length_km*b.parallel*zb),
              g=float(b.g_us_per_km*1e-6*b.length_km*b.parallel*zb))
    assert len(n.trafo) == 0 and len(n.shunt) == 0
    for idx, b in n.load.iterrows():
        lid=f'D{idx}'
        s.add('PQ', idx=lid, bus=int(b.bus)+1, Vn=float(n.bus.loc[b.bus,'vn_kv']),
              p0=float(b.p_mw*b.scaling*b.in_service/sb), q0=float(b.q_mvar*b.scaling*b.in_service/sb))
        if event == 'load_step':
            for src in ['Ppf','Qpf']:
                s.add('Alter', dict(model='PQ', dev=lid, src=src, t=event_time, method='*', amount=1.1))
    for idx,b in n.ext_grid.iterrows():
        s.add('Slack', idx=f'S{idx}', bus=int(b.bus)+1, Vn=float(n.bus.loc[b.bus,'vn_kv']),
              v0=float(b.vm_pu), a0=float(b.va_degree)*np.pi/180., Sn=sb,
              p0=float(n.res_ext_grid.loc[idx,'p_mw']/sb),
              q0=float(n.res_ext_grid.loc[idx,'q_mvar']/sb))
    for k,(idx,b) in enumerate(n.sgen.iterrows()):
        gen=f'G{idx}'; reg=f'R{idx}'; elec=f'E{idx}'
        vn=float(n.bus.loc[b.bus,'vn_kv']); sn=float(env.s_max[k])
        s.add('PV', idx=gen, bus=int(b.bus)+1, Vn=vn, Sn=sn,
              p0=float(b.p_mw/sb), q0=float(b.q_mvar/sb),
              v0=float(n.res_bus.loc[b.bus,'vm_pu']))
        s.add('REGCA1', idx=reg, bus=int(b.bus)+1, gen=gen, Sn=sn,
              Tg=.1*factor, Tfltr=.1*factor)
        s.add('REECA1', idx=elec, reg=reg, PFFLAG=0,VFLAG=1,QFLAG=0,PFLAG=0,PQFLAG=1,
              Trv=.02*factor,Tp=.02*factor,Tiq=.02*factor,Tpord=.02*factor,
              Imax=1.1,PMAX=sn/sb,QMax=sn/sb,QMin=-sn/sb,Vref0=float(n.res_bus.loc[b.bus,'vm_pu']),
              Ip1=1.1,Ip2=1.1,Ip3=1.1,Ip4=1.1,Iq1=1.1,Iq2=1.1,Iq3=1.1,Iq4=1.1)
        if event == 'q_step':
            # Reference case: a 10% move toward zero, within capacity at frozen P.
            s.add('Alter', dict(model='REECA1', dev=elec, src='qref0', t=event_time, method='*',amount=.9))
        if event == 'q_replay':
            limit=float(env.args.action_scale*np.sqrt(max(0.,sn*sn-float(b.p_mw)**2)))
            target=float(np.clip(env.round2_q_next[k],-limit,limit))
            s.add('Alter',dict(model='REECA1',dev=elec,src='qref0',t=event_time,method='=',amount=-target/sb))
    if (OUT/'pycode/round2_full_codegen_complete.txt').exists():
        s.undill(autogen_stale=False)
    else:
        # ANDES 1.10.1's generated package imports all registered models.
        # A partial prepare leaves a package that cannot be reloaded.
        s.prepare(quick=True, nomp=True)
        (OUT/'pycode/round2_full_codegen_complete.txt').write_text(andes.__version__)
    s.setup()
    return s


def run_one(args):
    started=time.time()
    env, prov = load_case(args.feeder,args.hour)
    suffix='_fullfine' if args.full_fine else ''
    dest=OUT/f'f{args.feeder}_h{args.hour}_{args.event}_tau{args.factor:g}_dt{args.step:g}_te{args.event_time:g}{suffix}'
    dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/'provenance.json',prov)
    s=convert(env,args.factor,args.event,args.event_time,args.horizon,args.step)
    ok=s.PFlow.run()
    error=float(np.max(np.abs(s.Bus.v.v-env.powergrid.res_bus.vm_pu.to_numpy())))
    qerror=float(np.max(np.abs(s.PV.q.v*env.powergrid.sn_mva-env.powergrid.sgen.q_mvar.to_numpy())))
    qrel=float(np.max(np.abs(s.PV.q.v*env.powergrid.sn_mva-env.powergrid.sgen.q_mvar.to_numpy())/env.s_max))
    check=dict(power_flow=bool(ok),voltage_alignment_max_pu=error,q_alignment_max_mvar=qerror,
               q_alignment_max_rated_pu=qrel, andes_version=andes.__version__,tau_factor=args.factor,integration_step_s=args.step,
               event=args.event,event_time_s=args.event_time,horizon_s=args.horizon)
    write_json(dest/'initialization.json',check)
    print('INITIALIZATION',json.dumps(check),flush=True)
    assert ok and error < 1e-7 and qrel < 1e-4
    try:
        s.TDS.init()
    except Exception as exc:
        tb=exc.__traceback__
        while tb:
            loc=tb.tb_frame.f_locals
            obj=loc.get('self')
            if hasattr(obj,'class_name'):
                print('INIT_FAILURE', obj.class_name, loc.get('name'), flush=True)
            tb=tb.tb_next
        raise
    check['tds_initialized']=bool(s.TDS.initialized)
    check['tds_init_ok']=bool(s.TDS.test_init())
    write_json(dest/'initialization.json',check)
    assert check['tds_init_ok']
    # Linearization concerns the continuous converter/network with held commands,
    # not a 180 s sampled learned-controller stability certificate.
    if s.EIG._pre_check():
        state_matrix=np.array(s.EIG.calc_As())
        mu,vectors=np.linalg.eig(state_matrix)
        write_json(dest/'state_names.json',list(s.EIG.x_name))
        np.savez_compressed(dest/'linearization.npz',state_matrix=state_matrix,eigenvectors=vectors)
        pd.DataFrame(dict(real_s_inv=mu.real,imag_s_inv=mu.imag)).to_csv(dest/'eigenvalues.csv',index=False)
        nonzero=mu[np.abs(mu)>1e-6]
        check.update(eigen_zero_count=int(np.sum(np.abs(mu)<=1e-6)),
                     spectral_abscissa_all=float(mu.real.max()),
                     spectral_abscissa_nonzero=float(nonzero.real.max()))
    if args.event_time>5 and not args.full_fine:
        # Larger steps only during the unforced equilibrium hold, checked below.
        s.TDS.config.tstep=.05
        s.TDS.config.tf=args.event_time-1.
        assert s.TDS.run()
        s.TDS.config.tstep=args.step
        s.TDS.config.tf=args.horizon
    result=s.TDS.run()
    ts=s.dae.ts
    v=ts.y[:,s.Bus.v.a]; q=ts.y[:,s.REGCA1.Qe.a]*env.powergrid.sn_mva
    p=ts.y[:,s.REGCA1.Pe.a]*env.powergrid.sn_mva
    ip=ts.y[:,s.REGCA1.Ipout.a]; iq=ts.y[:,s.REGCA1.Iqout_y.a]
    times=ts.t
    np.savez_compressed(dest/'trajectory.npz',time_s=times,voltage_pu=v,q_mvar=q,p_mw=p,
                        current_system_pu=np.hypot(ip,iq),states=ts.x)
    table=pd.DataFrame(dict(time_s=times,vmin_pu=v.min(1),vmax_pu=v.max(1),
                            q_total_mvar=q.sum(1),p_total_mw=p.sum(1)))
    table.to_csv(dest/'trajectory_summary.csv',index=False)
    check.update(completed=bool(result),last_time_s=float(times[-1]),runtime_s=time.time()-started,
                 voltage_min_pu=float(v.min()),voltage_max_pu=float(v.max()),
                 max_initial_derivative=float(np.abs(ts.x[1]-ts.x[0]).max()/float(times[1]-times[0])),
                 max_apparent_power_ratio=float((np.hypot(p,q)/env.s_max[None,:]).max()),
                 max_current_device_pu=float((np.hypot(ip,iq)*env.powergrid.sn_mva/env.s_max[None,:]).max()))
    before=times < args.event_time-1e-6
    check['pre_event_voltage_drift_max_pu']=float(np.abs(v[before]-v[0]).max())
    post=times >= args.event_time
    p0=env.powergrid.sgen.p_mw.to_numpy().copy()
    q0=env.powergrid.sgen.q_mvar.to_numpy().copy()
    qfinal=q0.copy()
    if args.event=='q_replay':
        limit=env.args.action_scale*np.sqrt(np.maximum(0,env.s_max**2-p0**2))
        qfinal=np.clip(env.round2_q_next,-limit,limit)
        check['reference_projection_max_mvar']=float(np.abs(qfinal-env.round2_q_next).max())
    elif args.event=='q_step':
        qfinal=.9*q0
    if args.event=='load_step':
        env.powergrid.load['p_mw']*=1.1
        env.powergrid.load['q_mvar']*=1.1
    env.powergrid.sgen['q_mvar']=qfinal
    pp.runpp(env.powergrid,calculate_voltage_angles=True,tolerance_mva=1e-10,max_iteration=50)
    vfinal=env.powergrid.res_bus.vm_pu.to_numpy()
    qtol=np.maximum(.02*np.abs(qfinal-q0),1e-4*env.s_max)
    vtol=np.maximum(.02*np.abs(vfinal-v[0]),1e-5)
    goodq=np.all(np.abs(q[post]-qfinal)<=qtol,axis=1)
    goodv=np.all(np.abs(v[post]-vfinal)<=vtol,axis=1)
    def settling(good):
        bad=np.flatnonzero(~good)
        at=0 if len(bad)==0 else int(bad[-1])+1
        return float(times[post][at]-args.event_time) if at<len(good) else None
    check.update(q_settling_s=settling(goodq),voltage_settling_s=settling(goodv),
                 endpoint_voltage_error_max_pu=float(np.abs(v[-1]-vfinal).max()),
                 endpoint_q_error_max_rated_pu=float((np.abs(q[-1]-qfinal)/env.s_max).max()),
                 inverter_current_limit_sample_fraction=float(np.mean(np.hypot(ip,iq)*env.powergrid.sn_mva/env.s_max[None,:]>=1.1-1e-5)),
                 initial_q_mvar=q0.tolist(),final_q_target_mvar=qfinal.tolist(),
                 final_v_target_pu=vfinal.tolist())
    write_json(dest/'metrics.json',check)
    print('RESULT',json.dumps({k:v for k,v in check.items() if not isinstance(v,list)}),flush=True)
    return check


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--feeder',type=int,default=33)
    parser.add_argument('--hour',type=int,default=12)
    parser.add_argument('--factor',type=float,default=1.)
    parser.add_argument('--step',type=float,default=.005)
    parser.add_argument('--event',choices=['q_step','q_replay','load_step','none'],default='q_replay')
    parser.add_argument('--event-time',type=float,default=180.)
    parser.add_argument('--horizon',type=float,default=185.)
    parser.add_argument('--full-fine',action='store_true')
    parser.add_argument('--batch',action='store_true')
    args=parser.parse_args()
    if args.batch:
        import copy
        matrix=[]
        for feeder in [33,141]:
            for hour in [12,18,21]:
                for event in ['q_replay','load_step']:
                    a=copy.copy(args);a.feeder=feeder;a.hour=hour;a.event=event
                    matrix.append(a)
            for event in ['q_replay','load_step']:
                for factor in [.5,2.]:
                    a=copy.copy(args);a.feeder=feeder;a.hour=12;a.event=event;a.factor=factor
                    matrix.append(a)
                a=copy.copy(args);a.feeder=feeder;a.hour=12;a.event=event;a.step=.0025
                matrix.append(a)
        a=copy.copy(args);a.feeder=33;a.hour=12;a.event='q_replay';a.full_fine=True
        matrix.append(a)
        write_json(OUT/'dynamic_matrix.json',[vars(a) for a in matrix])
        report=[]
        for a in matrix:
            try:
                report.append(dict(arguments=vars(a),result=run_one(a)))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                report.append(dict(arguments=vars(a),failure=str(exc),type=type(exc).__name__))
            write_json(OUT/'dynamic_batch_status.json',report)
    else:
        run_one(args)
