"""Independent numerical references and invariants for research metrics/policy."""
import pytest
np = pytest.importorskip('numpy')
pytest.importorskip('sklearn')
from sklearn.metrics import average_precision_score, hamming_loss, log_loss, roc_auc_score

from patchinsight.assessment import (extended_metrics, fit_policy, paired_group_bootstrap,
                                    selective_metrics, temperature_scale)


def test_metrics_against_independent_references():
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, (40, 5))
    p = rng.uniform(.01, .99, (40, 5))
    result = extended_metrics(y, p)
    assert result['hamming_loss'] == pytest.approx(hamming_loss(y, p >= .5))
    assert result['macro_average_precision'] == pytest.approx(average_precision_score(y, p, average='macro'))
    assert result['macro_roc_auc'] == pytest.approx(roc_auc_score(y, p, average='macro'))
    assert result['mean_binary_log_loss'] == pytest.approx(np.mean([
        log_loss(y[:, i], p[:, i], labels=[0, 1]) for i in range(5)]))
    assert result['mean_binary_brier'] == pytest.approx(np.mean((y-p)**2))
    for label in result['per_label'].values():
        assert sum(b['n'] for b in label['reliability']) == len(y)


def test_undefined_metrics_and_empty_acceptance():
    result = extended_metrics([[0]*5]*3, [[.1]*5]*3)
    assert result['macro_average_precision'] is None
    assert result['macro_roc_auc'] is None
    assert all(v['average_precision'] is None for v in result['per_label'].values())
    selected = selective_metrics([[0]*5], [[.5]*5], .9)
    assert selected['coverage'] == 0 and selected['error_on_accepted'] is None
    for scores in ([[float('nan')]*5], [[1.1]*5], [[.1]*4]):
        with pytest.raises(ValueError):
            extended_metrics([[0]*5], scores)


def test_temperature_cannot_change_half_threshold_predictions():
    p = np.array([[.01, .2, .5, .8, .99]])
    for t in (.25, 1, 4):
        assert np.array_equal(temperature_scale(p, t) >= .5, p >= .5)
    with pytest.raises(ValueError):
        temperature_scale(p, float('nan'))


def test_policy_reports_unattainable_target():
    y = [[0]*5, [1]*5]
    p = [[.6]*5, [.4]*5]  # all wrong, cannot meet the target
    policy = fit_policy(y, p)
    assert policy['confidence_cutoff'] is None
    assert fit_policy(y, p) == policy
    result = paired_group_bootstrap(y, p, p, ['a', 'b'], repeats=50)
    assert result['percentile_95'] == [0, 0]
