"""Strict fixed-date evaluation and trajectory artifact contract.

The protocol keeps safety metrics on the full planned denominator.  When a
controlled transition fails, that transition and every remaining point in the
same day receive a conservative safety score, while undefined physical values
remain missing rather than being silently averaged as zeros.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

POINTS_PER_DAY = 240
CADENCE_MINUTES = 3
SUBMISSION_ROOT_NAME = "q1_submission_v2_strict"
REVISION_ROOT_NAME = "revision_access_41572"
PHYSICAL_METRICS = (
    "percentage_of_lower_than_lower_v",
    "percentage_of_higher_than_upper_v",
    "average_voltage_deviation",
    "average_voltage",
    "max_voltage_rise_deviation",
    "max_voltage_drop_deviation",
    "q_loss",
    "total_line_loss",
    "voltage_deviation_loss",
    "voltage_violation_loss",
    "q_loss_weighted",
    "total_loss",
    "reward",
    "action_projection_ratio",
    "reactive_capacity_utilization",
    "reactive_saturation_ratio",
    "control_limit_saturation_ratio",
    "control_limit_utilization",
)
SAFETY_METRICS = ("vrate", "cr", "destroy")
IDENTIFIER_FIELDS = {
    "day_index",
    "day_position",
    "step_index",
    "timestamp",
    "executed",
}
INVERTER_VECTOR_FIELDS = (
    "inverter_p_mw",
    "inverter_q_mvar",
    "inverter_s_max_mva",
)
INVERTER_CAPABILITY_TOLERANCE = 1e-10


@dataclass(frozen=True)
class FixedTestResult:
    summary: dict[str, Any]
    step_rows: list[dict[str, Any]]
    daily_rows: list[dict[str, Any]]
    node_voltages: np.ndarray
    timestamps: np.ndarray
    executed_mask: np.ndarray


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _numeric(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _inverter_vector(value: Any, *, field: str) -> np.ndarray:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{field} must not be empty for an executed step")
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must be a JSON array or numeric vector") from exc
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a numeric vector") from exc
    if vector.ndim != 1:
        raise ValueError(f"{field} must be a one-dimensional inverter vector")
    if vector.size == 0:
        raise ValueError(f"{field} must contain at least one inverter")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{field} must contain only finite values")
    return vector.copy()


def _validated_inverter_vectors(source: dict[str, Any]) -> dict[str, np.ndarray]:
    missing = [field for field in INVERTER_VECTOR_FIELDS if field not in source]
    if missing:
        raise ValueError(
            "executed step is missing inverter evidence: " + ", ".join(missing)
        )
    vectors = {
        field: _inverter_vector(source[field], field=field)
        for field in INVERTER_VECTOR_FIELDS
    }
    lengths = {vector.size for vector in vectors.values()}
    if len(lengths) != 1:
        raise ValueError("inverter evidence vectors must have equal lengths")

    p_mw = vectors["inverter_p_mw"]
    q_mvar = vectors["inverter_q_mvar"]
    s_max_mva = vectors["inverter_s_max_mva"]
    if np.any(s_max_mva < 0.0):
        raise ValueError("inverter_s_max_mva must be non-negative")
    if np.any(np.abs(p_mw) > s_max_mva + INVERTER_CAPABILITY_TOLERANCE):
        raise ValueError("inverter active power exceeds the nameplate limit")
    try:
        with np.errstate(over="raise", invalid="raise"):
            capacity = np.sqrt(np.maximum(s_max_mva**2 - p_mw**2, 0.0))
    except FloatingPointError as exc:
        raise ValueError("inverter capability calculation must remain finite") from exc
    if not np.all(np.isfinite(capacity)):
        raise ValueError("inverter capability calculation must remain finite")
    if np.any(np.abs(q_mvar) > capacity + INVERTER_CAPABILITY_TOLERANCE):
        raise ValueError(
            "inverter_q_mvar exceeds the instantaneous inverter capability"
        )
    return vectors


def _validate_step_row_vectors(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        if bool(row.get("executed", False)):
            try:
                _validated_inverter_vectors(row)
            except ValueError as exc:
                raise ValueError(
                    f"invalid inverter evidence at step row {index}: {exc}"
                ) from exc
            continue
        for field in INVERTER_VECTOR_FIELDS:
            value = row.get(field)
            blank = value is None or (isinstance(value, str) and value == "")
            if not blank:
                raise ValueError(
                    f"failed step row {index} must leave {field} empty or missing"
                )


def _timestamp64(value: Any) -> np.datetime64:
    try:
        timestamp = np.datetime64(value, "ns")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"fixed-test timestamp is not datetime-like: {value!r}"
        ) from exc
    if np.isnat(timestamp):
        raise ValueError("fixed-test timestamp must not be NaT")
    return timestamp


def _fallback_timestamp(day: int, start_hour: int, step: int) -> np.datetime64:
    return (
        np.datetime64("2000-01-01T00:00:00", "ns")
        + np.timedelta64(int(day), "D")
        + np.timedelta64(int(start_hour), "h")
        + np.timedelta64(CADENCE_MINUTES * int(step), "m")
    )


def _current_timestamp(env: Any, day: int, start_hour: int, step: int) -> np.datetime64:
    if hasattr(env, "current_timestamp"):
        return _timestamp64(env.current_timestamp())
    return _fallback_timestamp(day, start_hour, step)


def _bus_snapshot(env: Any) -> tuple[list[Any], np.ndarray]:
    try:
        frame = env.powergrid.res_bus.sort_index()
        order = [_python_scalar(item) for item in frame.index.tolist()]
        voltage = frame["vm_pu"].to_numpy(dtype=float, copy=True)
    except (AttributeError, KeyError):
        return [], np.empty((0,), dtype=float)
    return order, voltage


@contextlib.contextmanager
def _force_zero_reset_action(env: Any):
    args = getattr(env, "args", None)
    if args is None:
        yield
        return

    if isinstance(args, tuple) and callable(getattr(args, "_replace", None)):
        env.args = args._replace(reset_action=False)
        try:
            yield
        finally:
            env.args = args
        return

    if isinstance(args, dict):
        present = "reset_action" in args
        previous = args.get("reset_action")
        args["reset_action"] = False
        try:
            yield
        finally:
            if present:
                args["reset_action"] = previous
            else:
                args.pop("reset_action", None)
            env.args = args
        return

    present = hasattr(args, "reset_action")
    previous = getattr(args, "reset_action", None)
    setattr(args, "reset_action", False)
    try:
        yield
    finally:
        if present:
            setattr(args, "reset_action", previous)
        else:
            delattr(args, "reset_action")
        env.args = args


def _executed_row(
    *,
    day: int,
    day_position: int,
    step: int,
    timestamp: np.datetime64,
    reward: float,
    info: dict[str, Any],
) -> dict[str, Any]:
    inverter_vectors = _validated_inverter_vectors(info)
    row: dict[str, Any] = {
        "day_index": int(day),
        "day_position": int(day_position),
        "step_index": int(step),
        "timestamp": np.datetime_as_string(timestamp, unit="s"),
        "executed": True,
        "vrate": _numeric(info.get("percentage_of_v_out_of_control")),
        "cr": _numeric(info.get("totally_controllable_ratio")),
        "destroy": _numeric(info.get("destroy"), 0.0),
        "percentage_of_v_out_of_control": _numeric(
            info.get("percentage_of_v_out_of_control")
        ),
        "totally_controllable_ratio": _numeric(info.get("totally_controllable_ratio")),
        "reward": _numeric(reward),
    }
    for key, value in info.items():
        if key not in row and key not in INVERTER_VECTOR_FIELDS:
            row[key] = _python_scalar(value)
    row.update(inverter_vectors)
    for key in PHYSICAL_METRICS:
        row.setdefault(key, float("nan"))
    return row


def _failure_row(
    *,
    day: int,
    day_position: int,
    step: int,
    timestamp: np.datetime64,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "day_index": int(day),
        "day_position": int(day_position),
        "step_index": int(step),
        "timestamp": np.datetime_as_string(timestamp, unit="s"),
        "executed": False,
        "vrate": 1.0,
        "cr": 0.0,
        "destroy": 1.0,
        "percentage_of_v_out_of_control": 1.0,
        "totally_controllable_ratio": 0.0,
    }
    row.update({key: float("nan") for key in PHYSICAL_METRICS})
    return row


def _has_finite_safety_metrics(info: dict[str, Any]) -> bool:
    required = (
        "percentage_of_v_out_of_control",
        "totally_controllable_ratio",
        "destroy",
    )
    return all(key in info and np.isfinite(_numeric(info[key])) for key in required)


def _mean_with_denominator(
    rows: list[dict[str, Any]], key: str, *, executed_only: bool
) -> tuple[float, int]:
    values: list[float] = []
    for row in rows:
        if executed_only and not bool(row["executed"]):
            continue
        value = _numeric(row.get(key))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), 0
    return float(np.mean(values)), len(values)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    planned = len(rows)
    executed = sum(bool(row["executed"]) for row in rows)
    summary: dict[str, Any] = {
        "planned_steps": int(planned),
        "executed_steps": int(executed),
        "failed_steps": int(planned - executed),
    }
    denominators: dict[str, int] = {}
    for key in SAFETY_METRICS:
        value, denominator = _mean_with_denominator(rows, key, executed_only=False)
        summary[key] = value
        denominators[key] = denominator
        if denominator != planned:
            raise AssertionError(
                f"safety metric {key} denominator {denominator} != {planned} planned steps"
            )

    summary["percentage_of_v_out_of_control"] = summary["vrate"]
    summary["totally_controllable_ratio"] = summary["cr"]
    denominators["percentage_of_v_out_of_control"] = denominators["vrate"]
    denominators["totally_controllable_ratio"] = denominators["cr"]

    metric_keys = set(PHYSICAL_METRICS)
    for row in rows:
        for key, value in row.items():
            if key in IDENTIFIER_FIELDS or key in SAFETY_METRICS:
                continue
            if key in INVERTER_VECTOR_FIELDS:
                continue
            if key in {
                "percentage_of_v_out_of_control",
                "totally_controllable_ratio",
            }:
                continue
            if isinstance(value, (int, float, np.number)):
                metric_keys.add(key)
    for key in sorted(metric_keys):
        value, denominator = _mean_with_denominator(rows, key, executed_only=True)
        output_key = "mean_reward" if key == "reward" else key
        summary[output_key] = value
        denominators[output_key] = denominator
    summary["metric_denominators"] = denominators
    return summary


def _validate_shape(
    days: list[int], max_steps: int, start_hour: int, strict_shape: bool
) -> None:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not days:
        raise ValueError("days must not be empty")
    if strict_shape and (len(days) != 30 or max_steps != POINTS_PER_DAY):
        raise ValueError("strict fixed tests require exactly 30 days x 240 points")
    if strict_shape and start_hour != 6:
        raise ValueError("strict fixed tests must start at 06:00")
    allowed_windows = {
        tuple(range(1000, 1030)),
        tuple(range(1030, 1060)),
    }
    if strict_shape and tuple(days) not in allowed_windows:
        raise ValueError(
            "strict fixed tests require the declared validation or test date window"
        )


def _evaluate_fixed_days(
    *,
    env: Any,
    days: list[int],
    max_steps: int,
    start_hour: int,
    transition_factory: Callable[
        [Any], Callable[[Any], tuple[float, bool, dict[str, Any]]]
    ],
    strict_shape: bool,
) -> FixedTestResult:
    days = [int(day) for day in days]
    _validate_shape(days, int(max_steps), int(start_hour), strict_shape)

    all_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    all_timestamps: list[np.datetime64] = []
    all_executed: list[bool] = []
    voltage_rows: list[np.ndarray] = []
    bus_order: list[Any] | None = None

    with _force_zero_reset_action(env):
        for day_position, day in enumerate(days):
            initialization_error = None
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    state, _ = env.manual_reset(day, start_hour, 0)
            except Exception as error:
                if not _is_known_powerflow_nonconvergence(error):
                    raise
                initialization_error = error
                state = None
            current_order, initial_voltage = _bus_snapshot(env)
            if initialization_error is not None and not current_order:
                current_order = [_python_scalar(i) for i in env.powergrid.bus.sort_index().index]
                initial_voltage = np.full(len(current_order), np.nan)
            if bus_order is None:
                bus_order = current_order
            elif current_order != bus_order:
                raise ValueError("bus order changed during fixed evaluation")
            bus_count = len(bus_order)
            if initial_voltage.shape != (bus_count,):
                raise ValueError("bus-voltage vector does not match declared bus order")

            first_timestamp = _current_timestamp(env, day, start_hour, 0)
            transition = transition_factory(state) if initialization_error is None else None
            day_rows: list[dict[str, Any]] = []
            failure_step: int | None = 0 if initialization_error is not None else None
            failure_reason = str(initialization_error) if initialization_error is not None else None

            for step in range(max_steps if initialization_error is None else 0):
                planned_timestamp = first_timestamp + np.timedelta64(
                    CADENCE_MINUTES * step, "m"
                )
                captured_timestamp = _current_timestamp(env, day, start_hour, step)
                if captured_timestamp != planned_timestamp:
                    raise ValueError(
                        "fixed-test environment timestamp does not follow the "
                        "declared three-minute cadence"
                    )

                try:
                    reward, done, info = transition(state)
                except Exception as error:
                    if not _is_known_powerflow_nonconvergence(error):
                        raise
                    reward, done, info = float('nan'), True, {'destroy': 1.0}
                    failure_reason = str(error)
                info = dict(info or {})
                failed = not _has_finite_safety_metrics(info) or (
                    _numeric(info.get("destroy"), 0.0) >= 0.5
                )
                if failed:
                    row = _failure_row(
                        day=day,
                        day_position=day_position,
                        step=step,
                        timestamp=captured_timestamp,
                    )
                    day_rows.append(row)
                    all_timestamps.append(captured_timestamp)
                    all_executed.append(False)
                    voltage_rows.append(np.full(bus_count, np.nan, dtype=float))
                    failure_step = step
                    break

                row = _executed_row(
                    day=day,
                    day_position=day_position,
                    step=step,
                    timestamp=captured_timestamp,
                    reward=reward,
                    info=info,
                )
                _, voltage = _bus_snapshot(env)
                if voltage.shape != (bus_count,):
                    raise ValueError(
                        "bus-voltage vector changed shape during evaluation"
                    )
                day_rows.append(row)
                all_timestamps.append(captured_timestamp)
                all_executed.append(True)
                voltage_rows.append(voltage)

                if done:
                    if step + 1 < max_steps:
                        failure_step = step + 1
                    break
                try:
                    state = env.get_obs()
                except Exception as error:
                    if not _is_known_powerflow_nonconvergence(error):
                        raise
                    # The current controlled snapshot is valid; only subsequent
                    # points are missing when the next pre-control flow fails.
                    failure_step = step + 1
                    failure_reason = str(error)
                    break

            fill_start = len(day_rows)
            if fill_start < max_steps:
                if failure_step is None:
                    failure_step = fill_start
                for step in range(fill_start, max_steps):
                    timestamp = first_timestamp + np.timedelta64(
                        CADENCE_MINUTES * step, "m"
                    )
                    row = _failure_row(
                        day=day,
                        day_position=day_position,
                        step=step,
                        timestamp=timestamp,
                    )
                    day_rows.append(row)
                    all_timestamps.append(timestamp)
                    all_executed.append(False)
                    voltage_rows.append(np.full(bus_count, np.nan, dtype=float))

            day_summary = _summarize_rows(day_rows)
            day_summary.update(
                {
                    "day_index": day,
                    "day_position": day_position,
                    "date": day_rows[0]["timestamp"].split("T", 1)[0],
                    "failure_step": failure_step,
                    "failure_reason": failure_reason,
                }
            )
            daily_rows.append(day_summary)
            all_rows.extend(day_rows)

    summary = _summarize_rows(all_rows)
    summary.update(
        {
            "planned_days": len(days),
            "completed_days": sum(row["failed_steps"] == 0 for row in daily_rows),
            "failed_days": sum(row["failed_steps"] > 0 for row in daily_rows),
            "start_day": days[0],
            "end_day": days[-1],
            "points_per_day": int(max_steps),
            "start_hour": int(start_hour),
            "cadence_minutes": CADENCE_MINUTES,
            "bus_order": bus_order or [],
            "physical_metrics_denominator": "executed finite transitions only",
            "safety_metrics_denominator": "all planned transitions",
        }
    )
    node_voltages = (
        np.vstack(voltage_rows)
        if voltage_rows
        else np.empty((0, len(bus_order or [])), dtype=float)
    )
    return FixedTestResult(
        summary=summary,
        step_rows=all_rows,
        daily_rows=daily_rows,
        node_voltages=node_voltages,
        timestamps=np.asarray(all_timestamps, dtype="datetime64[ns]"),
        executed_mask=np.asarray(all_executed, dtype=bool),
    )


def evaluate_policy_fixed_days(
    *,
    env: Any,
    behaviour_net: Any,
    args: Any,
    days: list[int],
    max_steps: int = POINTS_PER_DAY,
    start_hour: int = 6,
    _strict_shape: bool = True,
) -> FixedTestResult:
    """Evaluate a learned policy on deterministic, fixed-date trajectories."""
    import torch as th

    from utilities.util import prep_obs, translate_action

    device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")
    behaviour_net = behaviour_net.to(device)
    was_training = bool(behaviour_net.training)
    behaviour_net.eval()

    def transition_factory(_state):
        last_hid = behaviour_net.policy_dicts[0].init_hidden()

        def transition(state):
            nonlocal last_hid
            state_tensor = (
                prep_obs(state)
                .contiguous()
                .view(1, args.agent_num, args.obs_size)
                .to(device)
            )
            with th.no_grad():
                action, _, _, _, hid = behaviour_net.get_actions(
                    state_tensor,
                    status="test",
                    exploration=False,
                    actions_avail=th.tensor(env.get_avail_actions()),
                    target=False,
                    last_hid=last_hid,
                )
            _, actual = translate_action(args, action, env)
            reward, done, info = env.step(actual, add_noise=False)
            last_hid = hid
            return float(reward), bool(done), dict(info)

        return transition

    try:
        return _evaluate_fixed_days(
            env=env,
            days=days,
            max_steps=max_steps,
            start_hour=start_hour,
            transition_factory=transition_factory,
            strict_shape=_strict_shape,
        )
    finally:
        behaviour_net.train(was_training)


def _is_known_powerflow_nonconvergence(error: BaseException) -> bool:
    known_names = {"LoadflowNotConverged", "OPFNotConverged"}
    return any(
        cls.__name__ in known_names and cls.__module__.startswith("pandapower")
        for cls in type(error).__mro__
    )


def evaluate_controller_fixed_days(
    *,
    env: Any,
    controller: Callable[[Any], Any],
    days: list[int],
    max_steps: int = POINTS_PER_DAY,
    start_hour: int = 6,
    _strict_shape: bool = True,
) -> FixedTestResult:
    """Evaluate a controller through the same accounting engine as policies.

    The controller mutates the current network and returns only a boolean
    convergence flag.  This protocol exclusively owns reward recording, the
    deterministic no-noise time advance, and termination accounting.
    """

    def transition_factory(_state):
        def transition(_current_state):
            try:
                outcome = controller(env)
            except Exception as error:
                if _is_known_powerflow_nonconvergence(error):
                    outcome = False
                else:
                    raise
            if not isinstance(outcome, (bool, np.bool_)):
                raise TypeError("controller must return a boolean convergence flag")
            if not bool(outcome):
                return (
                    float("nan"),
                    True,
                    {
                        "destroy": 1.0,
                        "percentage_of_v_out_of_control": 1.0,
                        "totally_controllable_ratio": 0.0,
                    },
                )

            reward, info = env._calc_reward()
            env.steps += 1
            if hasattr(env, "sum_rewards"):
                env.sum_rewards += float(reward)
            done = env.steps >= env.episode_limit
            if not done:
                env._set_demand_and_pv(add_noise=False)
                if hasattr(env, "_loaded_step"):
                    env._loaded_step = env.steps
            return float(reward), bool(done), dict(info)

        return transition

    return _evaluate_fixed_days(
        env=env,
        days=days,
        max_steps=max_steps,
        start_hour=start_hour,
        transition_factory=transition_factory,
        strict_shape=_strict_shape,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if any("executed" in row for row in rows):
        _validate_step_row_vectors(rows)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=keys, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        encoded = dict(row)
        if "executed" in row:
            executed = bool(row["executed"])
            encoded["executed"] = "true" if executed else "false"
            if executed:
                vectors = _validated_inverter_vectors(row)
                for field, vector in vectors.items():
                    encoded[field] = json.dumps(
                        vector.tolist(),
                        allow_nan=False,
                        separators=(",", ":"),
                    )
            else:
                for field in INVERTER_VECTOR_FIELDS:
                    if field in keys:
                        encoded[field] = ""
        writer.writerow(encoded)
    return buffer.getvalue()


def _validate_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if not any(name in resolved.parts for name in (SUBMISSION_ROOT_NAME, REVISION_ROOT_NAME)):
        raise ValueError(
            f"fixed-test artifacts must be under {SUBMISSION_ROOT_NAME} or {REVISION_ROOT_NAME}"
        )
    if output_dir.exists():
        raise FileExistsError(
            f"refusing overwrite of fixed-test directory: {output_dir}"
        )


def _artifact_paths(directory: Path, stem: str) -> dict[str, Path]:
    return {
        "step_metrics": directory / f"{stem}_step_metrics.csv",
        "daily_metrics": directory / f"{stem}_daily_metrics.csv",
        "node_voltages": directory / f"{stem}_node_voltages.npz",
        "trajectory_schema": directory / f"{stem}_trajectory_schema.json",
        "summary": directory / f"{stem}_summary.json",
    }


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _validate_staged_artifacts(
    staging_dir: Path,
    staged: dict[str, Path],
    result: FixedTestResult,
) -> None:
    expected_names = {path.name for path in staged.values()}
    actual_names = {path.name for path in staging_dir.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError("staged fixed-test artifact set is incomplete")
    if _csv_row_count(staged["step_metrics"]) != len(result.step_rows):
        raise RuntimeError("staged step_metrics row count is invalid")
    with staged["step_metrics"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        persisted_rows = list(reader)
    if reader.fieldnames is None or reader.fieldnames.count("executed") != 1:
        raise RuntimeError(
            "staged step_metrics must contain exactly one executed column"
        )
    if result.executed_mask.shape != (len(result.step_rows),):
        raise RuntimeError("result execution mask must align one-to-one with step rows")
    for index, row in enumerate(persisted_rows):
        raw_executed = row["executed"]
        if raw_executed not in {"true", "false"}:
            raise RuntimeError(
                "staged step row "
                f"{index} has non-canonical executed value {raw_executed!r}"
            )
        executed = raw_executed == "true"
        original_executed = result.step_rows[index].get("executed")
        if not isinstance(original_executed, (bool, np.bool_)):
            raise RuntimeError(
                f"original step row {index} must contain a boolean executed value"
            )
        mask_executed = bool(result.executed_mask[index])
        if executed != bool(original_executed) or executed != mask_executed:
            raise RuntimeError(f"staged step row {index} execution state mismatch")
        if executed:
            try:
                _validated_inverter_vectors(row)
            except ValueError as exc:
                raise RuntimeError(
                    f"staged step row {index} has invalid inverter evidence: {exc}"
                ) from exc
        elif any(row.get(field, "").strip() for field in INVERTER_VECTOR_FIELDS):
            raise RuntimeError(
                f"staged failed step row {index} contains inverter evidence"
            )
    if _csv_row_count(staged["daily_metrics"]) != len(result.daily_rows):
        raise RuntimeError("staged daily_metrics row count is invalid")
    with np.load(staged["node_voltages"], allow_pickle=False) as arrays:
        if arrays["node_voltages"].shape != result.node_voltages.shape:
            raise RuntimeError("staged node-voltage shape is invalid")
        if arrays["timestamps"].shape != result.timestamps.shape:
            raise RuntimeError("staged timestamp shape is invalid")
        if arrays["executed_mask"].shape != result.executed_mask.shape:
            raise RuntimeError("staged execution-mask shape is invalid")
    schema = json.loads(staged["trajectory_schema"].read_text(encoding="utf-8"))
    summary = json.loads(staged["summary"].read_text(encoding="utf-8"))
    planned = int(result.summary["planned_steps"])
    if schema.get("planned_count") != planned:
        raise RuntimeError("staged schema planned count is invalid")
    if summary.get("planned_steps") != planned:
        raise RuntimeError("staged summary planned count is invalid")


def write_fixed_test_artifacts(
    result: FixedTestResult,
    output_dir: str | Path,
    stem: str,
) -> dict[str, str]:
    """Write a complete trajectory set without overwriting prior evidence."""
    output_dir = Path(output_dir)
    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a non-empty file-name component")
    _validate_step_row_vectors(result.step_rows)
    _validate_output_dir(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    targets = _artifact_paths(output_dir, stem)

    summary = result.summary
    schema = {
        "schema_version": 2,
        "units": {
            "node_voltages": "p.u.",
            "q_loss": "MVAr",
            "total_line_loss": "MW",
            "vrate": "fraction of buses outside voltage limits",
            "cr": "binary all-bus safety indicator averaged as a ratio",
        },
        "bus_order": summary.get("bus_order", []),
        "vector_fields": {
            "fields": list(INVERTER_VECTOR_FIELDS),
            "units": {
                "inverter_p_mw": "MW",
                "inverter_q_mvar": "MVAr",
                "inverter_s_max_mva": "MVA",
            },
            "encoding": "json-array-string",
            "capability_constraint": {
                "equation": (
                    "|p_i| <= S_i + 1e-10 and "
                    "|q_i| <= sqrt(max(S_i^2 - p_i^2, 0)) + 1e-10"
                ),
                "tolerance": INVERTER_CAPABILITY_TOLERANCE,
            },
        },
        "timestamp_convention": (
            "local source-data wall-clock captured immediately before each "
            "action; three-minute cadence"
        ),
        "date_window": {
            "start_day": summary.get("start_day"),
            "end_day": summary.get("end_day"),
            "day_count": summary.get("planned_days"),
        },
        "points_per_day": summary.get("points_per_day"),
        "failure_fill": {
            "vrate": 1.0,
            "cr": 0.0,
            "destroy": 1.0,
            "executed": False,
            "physical_metrics": "NaN",
            "node_voltages": "NaN",
            "scope": "failed transition and all remaining planned transitions in that day",
        },
        "planned_count": summary.get("planned_steps"),
        "executed_count": summary.get("executed_steps"),
        "failed_count": summary.get("failed_steps"),
        "metric_denominators": summary.get("metric_denominators", {}),
        "arrays": {
            "node_voltages": {
                "shape": list(result.node_voltages.shape),
                "dtype": str(result.node_voltages.dtype),
            },
            "timestamps": {
                "shape": list(result.timestamps.shape),
                "dtype": str(result.timestamps.dtype),
            },
            "executed_mask": {
                "shape": list(result.executed_mask.shape),
                "dtype": str(result.executed_mask.dtype),
            },
        },
    }

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    staged = _artifact_paths(staging_dir, stem)
    try:
        staged["step_metrics"].write_text(
            _csv_text(result.step_rows), encoding="utf-8", newline=""
        )
        staged["daily_metrics"].write_text(
            _csv_text(result.daily_rows), encoding="utf-8", newline=""
        )
        with staged["node_voltages"].open("wb") as handle:
            np.savez_compressed(
                handle,
                node_voltages=result.node_voltages,
                timestamps=result.timestamps,
                executed_mask=result.executed_mask,
                bus_order=np.asarray(summary.get("bus_order", [])),
            )
        staged["trajectory_schema"].write_text(
            json.dumps(_json_safe(schema), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        staged["summary"].write_text(
            json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _validate_staged_artifacts(staging_dir, staged, result)
        os.replace(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return {key: str(path) for key, path in targets.items()}
