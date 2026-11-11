"""Fixed-dispatch AC continuation; a static margin, never a dynamic proxy.

Pseudo-arclength predictor/corrector follows a solution branch through a
saddle-node. A failed Newton iteration is not called a collapse point.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.sparse import bmat, csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from pandapower.pypower.dSbus_dV import dSbus_dV
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pypower.idx_bus import VM,VA,BUS_TYPE,REF

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.round2_stability_dynamic import load_case, OUT, write_json


class ACPath:
    def __init__(self,env):
        net=env.powergrid
        ppc=net._ppc; it=ppc['internal']
        bus=it.get('bus',ppc['bus']); gen=it.get('gen',ppc['gen'])
        self.Y=it['Ybus']; self.n=len(bus); self.sb=ppc['baseMVA']
        self.Vbase=bus[:,VM]*np.exp(1j*bus[:,VA]*np.pi/180)
        self.ids=np.flatnonzero(bus[:,BUS_TYPE] != REF)
        self.S=makeSbus(self.sb,bus,gen)
        self.D=np.zeros(self.n,complex)
        lookup=net._pd2ppc_lookups['bus']
        for _,r in net.load.iterrows():
            self.D[lookup[int(r.bus)]] += r.scaling*r.in_service*(r.p_mw+1j*r.q_mvar)/self.sb
        self.x0=np.r_[np.angle(self.Vbase)[self.ids],np.abs(self.Vbase)[self.ids],0.]
        self.m=len(self.ids)

    def eval(self,x,jac=True):
        V=self.Vbase.copy(); ids=self.ids; m=self.m
        V[ids]=x[m:2*m]*np.exp(1j*x[:m])
        mis=V*np.conj(self.Y@V)-self.S+x[-1]*self.D
        f=np.r_[mis[ids].real,mis[ids].imag]
        if not jac:
            return f,V
        dvm,dva=dSbus_dV(self.Y,V)
        J=bmat([[dva[ids,:][:,ids].real,dvm[ids,:][:,ids].real],
                [dva[ids,:][:,ids].imag,dvm[ids,:][:,ids].imag]],format='csr')
        D=np.r_[self.D[ids].real,self.D[ids].imag]
        return f,V,bmat([[J,csr_matrix(D[:,None])]],format='csr')

    def trace(self,ds=.04,max_points=1200):
        x=self.x0.copy(); prev=np.zeros_like(x); prev[-1]=1.
        rows=[]; states=[]; turn=False; failure=None
        f,v,J=self.eval(x)
        rng=np.random.default_rng(0); d=rng.standard_normal(len(x)); d/=np.linalg.norm(d)
        fd=(self.eval(x+1e-6*d,False)[0]-self.eval(x-1e-6*d,False)[0])/2e-6
        derivative_error=float(np.max(np.abs(fd-J@d)))
        derivative_relative_error=derivative_error/max(1.,float(np.max(np.abs(J@d))))
        print('DERIVATIVE_CHECK',derivative_error,derivative_relative_error,float(np.max(np.abs(f))),flush=True)
        assert derivative_relative_error < 1e-7 and np.max(np.abs(f))<1e-7
        for k in range(max_points):
            f,v,J=self.eval(x)
            border=bmat([[J],[csr_matrix(prev[None,:])]],format='csc')
            rhs=np.zeros_like(x); rhs[-1]=1.
            t=spsolve(border,rhs); t/=np.linalg.norm(t)
            if t@prev<0: t=-t
            rows.append(dict(point=k,additional_load_fraction=x[-1],vmin_pu=np.abs(v).min(),
                             vmax_pu=np.abs(v).max(),lambda_tangent=t[-1],
                             ac_residual_pu=float(np.abs(f).max())))
            states.append(x.copy())
            if k>0 and t[-1]<0 and rows[-2]['lambda_tangent']>=0:
                turn=True
            if turn and len(rows)>4 and x[-1]<max(r['additional_load_fraction'] for r in rows)-.03:
                break
            step=ds; success=False
            for retry in range(8):
                pred=x+step*t; z=pred.copy()
                for _ in range(15):
                    ff,_,jj=self.eval(z)
                    aug=np.r_[ff,(z-pred)@t]
                    if np.max(np.abs(aug))<1e-9:
                        success=True; break
                    zz=spsolve(bmat([[jj],[csr_matrix(t[None,:])]],format='csc'),-aug)
                    z+=zz
                    if not np.all(np.isfinite(z)) or np.min(z[self.m:2*self.m])<.05:
                        break
                if success: break
                step/=2
            if not success:
                failure=f'corrector failed after step reduction, point={k}'
                break
            x=z;prev=t
        df=pd.DataFrame(rows)
        peak=int(np.argmax(df.additional_load_fraction)); summary=dict(
            branch_turn_detected=turn,not_convergence_failure_as_collapse=True,
            failure=failure,derivative_max_abs_error=derivative_error,
            derivative_max_relative_error=derivative_relative_error,
            arc_step=ds,points=len(df),initial_vmin_pu=float(df.iloc[0].vmin_pu),
            initial_vmax_pu=float(df.iloc[0].vmax_pu),
            fold_load_factor_estimate=float(1+df.iloc[peak].additional_load_fraction) if turn else None,
            fold_vmin_pu=float(df.iloc[peak].vmin_pu) if turn else None,
            max_ac_residual_pu=float(df.ac_residual_pu.max()))
        return df,np.array(states),summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--feeder',type=int,default=33)
    parser.add_argument('--hour',type=int,default=12); parser.add_argument('--step',type=float,default=.04)
    parser.add_argument('--zero-q',action='store_true'); a=parser.parse_args()
    env,prov=load_case(a.feeder,a.hour)
    if a.zero_q:
        import pandapower as pp
        env.powergrid.sgen['q_mvar']=0.
        pp.runpp(env.powergrid,calculate_voltage_angles=True,tolerance_mva=1e-10,max_iteration=50)
    dest=OUT/'static'/f'f{a.feeder}_h{a.hour}_{"zeroq" if a.zero_q else "policy"}_ds{a.step:g}'
    dest.mkdir(parents=True,exist_ok=True)
    path=ACPath(env);df,x,summary=path.trace(a.step)
    df.to_csv(dest/'continuation.csv',index=False);np.savez_compressed(dest/'states.npz',state=x)
    summary.update(feeder=a.feeder,hour=a.hour,dispatch='zero_q' if a.zero_q else 'recorded_policy',
                   direction='All original PQ loads increase proportionally at fixed power factor; PV P/Q held.')
    write_json(dest/'metrics.json',summary);write_json(dest/'provenance.json',prov)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    if '--batch' in sys.argv:
        for feeder in [33,141]:
            for hour in [12,18,21]:
                for zero in [False,True]:
                    sys.argv=[sys.argv[0],'--feeder',str(feeder),'--hour',str(hour)]
                    if zero: sys.argv+=['--zero-q']
                    main()
            sys.argv=[sys.argv[0],'--feeder',str(feeder),'--hour','12','--step','.02']
            main()
    else:
        main()
