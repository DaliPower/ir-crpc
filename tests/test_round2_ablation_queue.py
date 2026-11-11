"""Dispatch controls change concurrency/order only, never registered runs."""
import json
import pytest
from experiments.round2_ablation import build_plan
from experiments.round2_ablation_queue import schedule_runs, read_control


def test_new_first_interleaves_networks_without_mutating_matrix():
    plan=build_plan('formal')
    original=list(plan['runs'])
    plan['reuse_runs']={r['run_id']:{} for r in original if r['round2_variant']=='A0'}
    runs=schedule_runs(plan,original)
    assert plan['runs']==original
    assert {r['run_id'] for r in runs}=={r['run_id'] for r in original}
    assert all(r['round2_variant']!='A0' for r in runs[:100])
    assert [r['scenario'] for r in runs[:4]]==[
        'case33_3min_final','case141_3min_final','case33_3min_final','case141_3min_final']


def test_control_allows_live_four_to_eight_and_pause(tmp_path):
    p=tmp_path/'control.json'
    p.write_text(json.dumps(dict(workers=8,pause_dispatch=False)))
    assert read_control(p)==dict(workers=8,pause_dispatch=False)
    p.write_text(json.dumps(dict(workers=4,pause_dispatch=True)))
    assert read_control(p)==dict(workers=4,pause_dispatch=True)


@pytest.mark.parametrize('control',[
    dict(workers=0,pause_dispatch=False),dict(workers=9,pause_dispatch=False),
    dict(workers=1.5,pause_dispatch=False),dict(workers=True,pause_dispatch=False),
    dict(workers=4,pause_dispatch='false')])
def test_invalid_control_cannot_expand_concurrency(tmp_path,control):
    p=tmp_path/'control.json';p.write_text(json.dumps(control))
    with pytest.raises(ValueError):read_control(p)
