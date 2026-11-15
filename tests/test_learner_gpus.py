"""``rollout.learner_gpus``: one update over several processes, here two on the cpu over gloo.

The claim the setting makes is that the minibatches' gradients add up to the batch's wherever
they are computed. The strongest test of it is the whole update both ways: one process, then two
from the same seed, and the weights after the update agree to float rounding. Under Adam a
gradient at the level of rounding can take a whole step either way, so both runs here step with
SGD, whose step is the gradient; rank 0 is the only optimizer, so only its factory matters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import pytest
import torch

from royalelearn.config import check_consistency
from royalelearn.errors import PreflightError
from royalelearn.learn import ddp
from royalelearn.learn import ppo as ppo_module
from royalelearn.testing import coordinator, tiny_config


def _sgd(params: Any, **kwargs: Any) -> torch.optim.Optimizer:
    return torch.optim.SGD(params, lr=kwargs["lr"])


def _config(tmp_path: Path, learners: int | None, **ppo: Any) -> Any:
    base = tiny_config(tmp_path)
    rollout = base.rollout
    if learners is not None:
        rollout = msgspec.structs.replace(rollout, learner_gpus=learners)
    # Learning rates large enough that the update moves the weights well clear of rounding.
    fields = {"lr_actor": 1e-2, "lr_critic": 1e-2, **ppo}
    return tiny_config(
        tmp_path, rollout=rollout, ppo=msgspec.structs.replace(base.ppo, **fields)
    )


def _weights_after_one_iteration(config: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    with coordinator(config) as run:
        before = {k: v.detach().clone() for k, v in run.model.state_dict().items()}
        run.iterate()
        after = {k: v.detach().clone() for k, v in run.model.state_dict().items()}
        row = dict(run.rows[-1])
    moved = max(float((after[k] - before[k]).abs().max()) for k in after)
    assert moved > 1e-4, "the update did not move the weights, so agreeing says nothing"
    return after, row


@pytest.fixture
def cpu_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ddp, "ALLOW_CPU_GROUPS", True)
    monkeypatch.setattr(ddp, "COLLECTIVE_TIMEOUT_S", 60.0)
    monkeypatch.setattr(ppo_module, "_adam", _sgd)


@pytest.mark.parametrize("forced_rows", ["all", "critic_only"])
def test_two_learners_take_the_step_one_takes(
    tmp_path: Path, cpu_groups: None, forced_rows: str
) -> None:
    """Plant: weight a rank's minibatches by their share of the rank's part instead of the
    whole batch, or drop the gradient sum, and the two disagree by the size of the update."""
    one, _ = _weights_after_one_iteration(
        _config(tmp_path / "one", None, forced_rows=forced_rows)
    )
    two, row = _weights_after_one_iteration(
        _config(tmp_path / "two", 2, forced_rows=forced_rows)
    )
    assert one.keys() == two.keys()
    worst = max(float((one[k] - two[k]).abs().max()) for k in one)
    assert worst < 1e-5, worst
    # The seconds between the ranks are their own phase.
    assert float(row["time/update_sync"]) > 0.0


def test_two_learners_under_overlap_take_the_steps_one_takes(
    tmp_path: Path, cpu_groups: None
) -> None:
    """With ``rollout.overlap`` the second batch is collected by the actor as it stood before
    the first update, and rank 0 checks the ratio against that snapshot; the helpers train."""

    def weights(folder: Path, learners: int | None) -> dict[str, torch.Tensor]:
        config = _config(folder, learners)
        config = msgspec.structs.replace(
            config, rollout=msgspec.structs.replace(config.rollout, overlap=True)
        )
        with coordinator(config) as run:
            run.learn(until_timesteps=2 * config.ppo.timesteps_per_iteration)
            assert len(run.rows) == 2
            return {k: v.detach().clone() for k, v in run.model.state_dict().items()}

    one, two = weights(tmp_path / "one", None), weights(tmp_path / "two", 2)
    worst = max(float((one[k] - two[k]).abs().max()) for k in one)
    assert worst < 1e-5, worst


def test_a_resume_onto_two_learners_trains_from_the_checkpoint(
    tmp_path: Path, cpu_groups: None
) -> None:
    """A run checkpointed on one learner, resumed on two and on one: the same next step.

    Plant: leave the helpers on the weights they were built with instead of rank 0's, and the
    step on two learners is taken from two different points.
    """
    with coordinator(_config(tmp_path, None)) as first:
        first.iterate()
        saved = first.checkpoint()
        run_dir = first.run_dir

    def resumed(learners: int | None) -> dict[str, torch.Tensor]:
        with coordinator(_config(tmp_path, learners), resume=saved, run_dir=run_dir) as run:
            run.iterate()
            return {k: v.detach().clone() for k, v in run.model.state_dict().items()}

    one, two = resumed(None), resumed(2)
    worst = max(float((one[k] - two[k]).abs().max()) for k in one)
    assert worst < 1e-5, worst


def test_a_helper_that_trains_on_other_rows_is_caught(
    tmp_path: Path, cpu_groups: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A debug iteration gathers each helper's first minibatch again on rank 0 and compares."""
    import ddp_support

    monkeypatch.setattr(ddp, "RunRecipe", ddp_support.ShiftedAdvantages)
    with (
        coordinator(_config(tmp_path, 2)) as run,
        pytest.raises(AssertionError, match=r"learner rank 1 .*advantages"),
    ):
        run.iterate()


def test_a_helper_that_fails_stops_the_update_with_its_reason(
    tmp_path: Path, cpu_groups: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ddp_support

    monkeypatch.setattr(ddp, "RunRecipe", ddp_support.FailsOnUpdate)
    with (
        coordinator(_config(tmp_path, 2)) as run,
        pytest.raises(ddp.HelperFailure, match="planted"),
    ):
        run.iterate()


def test_the_setting_is_refused_where_it_cannot_mean_anything(tmp_path: Path) -> None:
    exact = _config(tmp_path, 2)
    exact = msgspec.structs.replace(
        exact, determinism=msgspec.structs.replace(exact.determinism, tier="run_exact")
    )
    assert any("run_exact" in problem for problem in check_consistency(exact))
    assert any("at least 1" in problem for problem in check_consistency(_config(tmp_path, 0)))
    # On the cpu, outside the tests' own hook: the setting is about GPUs.
    with (
        pytest.raises(PreflightError, match="spreads the update over GPUs"),
        coordinator(_config(tmp_path, 2)),
    ):
        pass
