"""Planted helper ranks for ``tests/test_learner_gpus.py``.

Here rather than in the test module because a helper rank is a spawned process that unpickles
its recipe by module name, and a test module is not something it should import.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from royalelearn.learn.ddp import RunRecipe


class ShiftedAdvantages(RunRecipe):
    """A helper whose view of the rectangle has every advantage moved by one: it trains on rows
    rank 0 would not have handed it."""

    def build(self, **kwargs: Any) -> Any:
        replica = super().build(**kwargs)
        buffer = replica.buffer
        take = buffer.take_replica_columns

        def shifted(iteration: int, cycles: int, columns: Any) -> None:
            take(iteration, cycles, columns)
            buffer.advantage += np.float32(1.0)

        buffer.take_replica_columns = shifted  # type: ignore[method-assign]
        return replica


class FailsOnUpdate(RunRecipe):
    """A helper that raises when it is handed its first iteration."""

    def build(self, **kwargs: Any) -> Any:
        replica = super().build(**kwargs)

        def fail(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("planted: this helper cannot take its iteration")

        replica.buffer.take_replica_columns = fail  # type: ignore[method-assign]
        return replica


class SlowSecondHelper(RunRecipe):
    """Rank 2 builds for longer than one collective may wait, rank 1 at once: the ranks must not
    start their rendezvous until the slowest has built."""

    #: Seconds rank 2's build takes beyond the real one.
    DELAY_S = 15.0

    def build(self, **kwargs: Any) -> Any:
        import time

        if kwargs.get("rank") == 2:
            time.sleep(self.DELAY_S)
        return super().build(**kwargs)
