from types import SimpleNamespace
import numpy as np

from utilities.trainer import PGTrainer


def test_scalar_logger_skips_vector_evidence_without_losing_scalar_metrics():
    logged = []
    def add_scalar(name, value, step):
        assert np.asarray(value).ndim == 0, 'vectors are not TensorBoard scalars'
        logged.append((name, value, step))
    trainer = object.__new__(PGTrainer)
    trainer.logger = SimpleNamespace(add_scalar=add_scalar)
    trainer.episodes = 1
    trainer.logging({'mean_train_reward': -0.3, 'inverter_q_mvar': np.array([0.1, 0.2])})
    assert logged == [('data/mean_train_reward', -0.3, 1)]


def test_json_preserves_inverter_vectors_as_numbers():
    from experiments.run_revision_pilot import finite_json
    assert finite_json({'q': np.array([1.0, np.nan])}) == {'q': [1.0, None]}
