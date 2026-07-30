"""Rate throttle (SPEC §8 S5a `--rate` files/min)."""

from __future__ import annotations

from imsg.backfill.throttle import RateThrottle


def test_first_call_never_sleeps() -> None:
    sleeps: list[float] = []
    clock = iter([0.0])
    throttle = RateThrottle(60.0, sleep_fn=sleeps.append, clock_fn=lambda: next(clock))
    throttle.wait()
    assert sleeps == []


def test_second_call_sleeps_the_remaining_interval() -> None:
    sleeps: list[float] = []
    # 60/min -> 1s interval. First call at t=0, second call at t=0.4 -> sleep 0.6.
    clock_values = iter([0.0, 0.4, 1.0])
    throttle = RateThrottle(60.0, sleep_fn=sleeps.append, clock_fn=lambda: next(clock_values))
    throttle.wait()
    throttle.wait()
    assert sleeps == [0.6]


def test_no_sleep_when_already_slower_than_the_rate() -> None:
    sleeps: list[float] = []
    clock_values = iter([0.0, 5.0])
    throttle = RateThrottle(60.0, sleep_fn=sleeps.append, clock_fn=lambda: next(clock_values))
    throttle.wait()
    throttle.wait()
    assert sleeps == []


def test_invalid_rate_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="files_per_minute"):
        RateThrottle(0)
