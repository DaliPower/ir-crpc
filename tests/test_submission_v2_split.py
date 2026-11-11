import pytest

from experiments.run_risk_aware_multiseed import build_day_split


def test_submission_v2_split_is_strictly_forward():
    split = build_day_split(
        total_days=1097,
        validation_start_day=1000,
        validation_days=30,
        test_start_day=1030,
        test_days=30,
        guard_days=1,
    )

    assert split["training_days"] == list(range(0, 999))
    assert split["guard_days"] == [999]
    assert split["validation_days"] == list(range(1000, 1030))
    assert split["test_days"] == list(range(1030, 1060))
    assert split["unused_days"] == list(range(1060, 1097))


@pytest.mark.parametrize(
    ("kwargs", "boundary"),
    [
        ({"test_start_day": 1029}, "test_start_day"),
        ({"guard_days": 0}, "guard_days"),
        ({"test_start_day": 1031}, "test_start_day"),
        ({"test_days": 68}, "test_days"),
    ],
    ids=["overlap", "missing-guard", "gap-between-validation-and-test", "test-beyond-total-days"],
)
def test_submission_v2_split_rejects_invalid_boundaries(kwargs, boundary):
    options = {
        "total_days": 1097,
        "validation_start_day": 1000,
        "validation_days": 30,
        "test_start_day": 1030,
        "test_days": 30,
        "guard_days": 1,
    }
    options.update(kwargs)

    with pytest.raises(ValueError, match=boundary):
        build_day_split(**options)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"validation_start_day": 900, "test_start_day": 930},
        {"validation_days": 31, "test_start_day": 1031},
        {"test_days": 31},
    ],
    ids=["shifted-forward-window", "validation-and-test-boundary-drift", "test-length-drift"],
)
def test_submission_v2_split_rejects_noncanonical_submission_boundaries(
    kwargs,
):
    options = {
        "total_days": 1097,
        "validation_start_day": 1000,
        "validation_days": 30,
        "test_start_day": 1030,
        "test_days": 30,
        "guard_days": 1,
    }
    options.update(kwargs)

    with pytest.raises(ValueError, match="submission boundaries"):
        build_day_split(**options)
