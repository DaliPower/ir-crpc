"""Load both real submitted seed-0 weights; compare actual validation observations."""
from pathlib import Path
import contextlib
import io
import json

import pytest
import torch

from models.matd3_reference import MATD3Reference
from models.matd3_round2_ablation import MATD3Round2Ablation
from experiments.round2_ablation import BASE_RUNS,SCENARIOS
from experiments.run_q1_ml_compare import make_env
from utilities.util import convert,prep_obs


@pytest.mark.parametrize('scenario',SCENARIOS)
def test_saved_a0_weights_match_on_real_validation_state(scenario):
    directory=BASE_RUNS/f'reference_B2__{scenario}__seed0'
    manifest=json.loads((directory/'manifest.json').read_text())
    cfg=json.loads((directory/'effective_configuration.json').read_text())['args']
    cfg.update(cuda=False)
    original_args=convert(cfg)
    original=MATD3Reference(original_args,MATD3Reference(original_args))
    cfg.update(round2_variant='A0')
    round_args=convert(cfg)
    new=MATD3Round2Ablation(round_args,MATD3Round2Ablation(round_args))
    weights=torch.load(directory/'best_model.pt',map_location='cpu',weights_only=False)['model_state_dict']
    original.load_state_dict(weights,strict=True);new.load_state_dict(weights,strict=True)
    original.eval();new.eval()
    config=dict(manifest['validation_environment'])
    config.update(start_day_candidates=[1000],add_noise=False)
    with contextlib.redirect_stdout(io.StringIO()):env=make_env(config)
    try:
        with contextlib.redirect_stdout(io.StringIO()):env.reset()
        obs=torch.as_tensor(env.get_obs(),dtype=torch.float32).unsqueeze(0)
        mask=torch.as_tensor(env.get_avail_actions(),dtype=torch.float32).unsqueeze(0)
        hidden=obs.new_zeros(1,original.n_,original.hid_dim)
        with torch.no_grad():
            a=original.get_actions(obs,'test',False,mask,last_hid=hidden)
            b=new.get_actions(obs,'test',False,mask,last_hid=hidden)
            assert torch.equal(a[1],b[1])
            assert torch.equal(a[4],b[4])
            assert torch.equal(original.value(obs,a[1]),new.value(obs,b[1]))
    finally:env.close()
