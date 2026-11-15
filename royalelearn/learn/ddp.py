"""``rollout.learner_gpus``: one update over several GPUs of one machine.

The coordinator's process is rank 0 and keeps everything it had: collection, inference, the
ladder, the gates, checkpoints and metric rows, the critic's pass over the rectangle, GAE, and
both optimizers. It spawns one helper process per further GPU itself (torch.multiprocessing,
spawned; no torchrun, so the launch line does not change). A helper builds the same actor-critic
and the same update from the config, attaches to the training rectangle's shared segment, and is
handed each iteration's scalar columns, schedule and term states over a queue.

Every rank walks the same batches, because the permutation is a function of the iteration and
the epoch. Each trainable cell belongs to one rank for the whole iteration, so a rank trains the
same cells in every epoch and a term that caches a frozen forward by cell in the first epoch
finds it in the later ones. A rank trains its cells of each batch in minibatches of the usual
size, each weighted by its share of the WHOLE batch, and a minibatch's loss is a mean scaled by
that weight or a sum scaled by a number decided per batch. The minibatches' gradients therefore
add up to the batch's wherever they were computed. At the end of each batch the gradients are
summed onto rank 0, which clips and steps, and the parameters are broadcast back. Rank 0 is the
only optimizer: there is no optimizer state to keep in step, and a checkpoint is rank 0's alone.

Rank 0's alone, and so a sample of one minibatch in ``world``: the update's diagnostics
(``ppo/kl`` and the rest), the ratio invariant, and what the terms measure. On a debug
iteration each helper's first minibatch is gathered again on rank 0 and the two are compared,
which is the check that a helper trains on the rows rank 0 would have.
"""

from __future__ import annotations

import contextlib
import datetime
import os
import queue as queue_module
import shutil
import tempfile
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import msgspec
import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from .buffer import Batch, Minibatch, RectBuffer
    from .ppo import PPOUpdate

__all__ = [
    "COLLECTIVE_TIMEOUT_S",
    "HelperFailure",
    "LearnerGroup",
    "Peers",
    "Replica",
    "ReplicaRecipe",
    "RunRecipe",
    "RunReplica",
    "learner_devices",
    "minibatch_digest",
]

#: Seconds one collective waits for a peer before the run stops. A collective waits for at most
#: a batch of minibatches on the slowest rank: the long phases of an iteration (collection, the
#: critic's pass, the gate) happen while the helpers wait on their queue, not in a collective.
COLLECTIVE_TIMEOUT_S = 300.0
#: Seconds a helper has to build its replica and say so: an import of torch, a model, the terms.
STARTUP_TIMEOUT_S = 600.0
#: Seconds rank 0 waits for the helpers to finish an update after its own last batch.
FINISH_TIMEOUT_S = 300.0
#: Seconds a stopping helper has to exit before it is terminated.
STOP_TIMEOUT_S = 30.0

#: Tests only: a group of processes on the cpu over gloo, which the coordinator otherwise
#: refuses, because the point of the setting is GPUs.
ALLOW_CPU_GROUPS = False


class HelperFailure(RuntimeError):
    """A helper rank stopped: what it said, or how it exited."""


class Replica(Protocol):
    """What a helper rank holds: its update and its view of the training rectangle."""

    update: PPOUpdate
    buffer: RectBuffer

    def champion(self, gated: bool = False) -> str | None: ...

    def set_champions(self, champions: tuple[str | None, str | None]) -> None: ...

    def close(self) -> None: ...


class ReplicaRecipe(Protocol):
    """Picklable: what a helper builds its replica from, in its own process."""

    def build(self, *, rank: int, world: int, device: torch.device) -> Replica: ...


def learner_devices(device: Any, world: int) -> list[str]:
    """The device of each rank: rank 0's own, then the next visible GPUs in turn."""
    device = torch.device(device)
    if device.type != "cuda":
        return [str(device)] * world
    count = torch.cuda.device_count()
    first = device.index if device.index is not None else torch.cuda.current_device()
    return [f"cuda:{(first + rank) % count}" for rank in range(world)]


def backend_for(device: Any) -> str:
    return "nccl" if torch.device(device).type == "cuda" else "gloo"


class Peers:
    """The collectives every rank runs at the same points of an update."""

    def __init__(self, rank: int, world: int) -> None:
        self.rank = int(rank)
        self.world = int(world)

    def reduce_gradients(self, params: Sequence[Parameter]) -> None:
        """Sum every rank's gradients onto rank 0.

        A parameter no rank gave a gradient keeps None on rank 0, as it would on one GPU, where
        Adam skips it; a zero in its place would decay its moments and move it.
        """
        import torch.distributed as dist

        if not params:
            return
        dtype = _one_dtype(params)
        device = params[0].device
        pieces = [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).to(dtype)
            for p in params
        ]
        present = torch.tensor(
            [float(p.grad is not None) for p in params], dtype=dtype, device=device
        )
        flat = torch.cat([*pieces, present])
        dist.reduce(flat, dst=0, op=dist.ReduceOp.SUM)
        if self.rank != 0:
            return
        sizes = [p.numel() for p in params]
        total = sum(sizes)
        had = flat[total:].cpu().tolist()
        for parameter, piece, count in zip(params, flat[:total].split(sizes), had, strict=True):
            if count > 0:
                parameter.grad = piece.view_as(parameter).to(parameter.dtype).clone()
            else:
                parameter.grad = None

    def broadcast_parameters(self, params: Sequence[Parameter]) -> None:
        """Rank 0's parameters, after its step, onto every rank."""
        import torch.distributed as dist

        if not params:
            return
        dtype = _one_dtype(params)
        sizes = [p.numel() for p in params]
        if self.rank == 0:
            flat = torch.cat([p.detach().reshape(-1).to(dtype) for p in params])
        else:
            flat = torch.empty(sum(sizes), dtype=dtype, device=params[0].device)
        dist.broadcast(flat, src=0)
        if self.rank == 0:
            return
        with torch.no_grad():
            for parameter, piece in zip(params, flat.split(sizes), strict=True):
                parameter.copy_(piece.view_as(parameter))

    def broadcast_model(self, model: torch.nn.Module) -> None:
        """Every tensor of rank 0's state dict onto every rank, at the start of an update.

        This is what keeps a helper's weights rank 0's through everything that changes them
        outside the update: a resume, a rollback, starting weights. In place, through the state
        dict's tensors, which share their storage with the parameters and buffers.
        """
        import torch.distributed as dist

        with torch.no_grad():
            for tensor in model.state_dict().values():
                if tensor.is_contiguous():
                    dist.broadcast(tensor, src=0)
                    continue
                staged = tensor.contiguous()
                dist.broadcast(staged, src=0)
                tensor.copy_(staged)


def _one_dtype(params: Sequence[Parameter]) -> torch.dtype:
    dtypes = {p.dtype for p in params}
    if len(dtypes) != 1:
        raise TypeError(
            f"rollout.learner_gpus sums the gradients in one buffer and these parameters are "
            f"{sorted(str(d) for d in dtypes)}"
        )
    return dtypes.pop()


def minibatch_digest(minibatch: Minibatch) -> dict[str, Any]:
    """Position-weighted float64 sums of everything a minibatch carries, on the host.

    Equal for two minibatches gathered from the same bytes and the same columns; a row moved,
    dropped or decoded differently changes the sum it is in. On the host in float64, so the
    comparison does not depend on which device reduced what.
    """
    digest: dict[str, Any] = {"cells": minibatch.cells.cpu().numpy().tolist()}

    def weighted(name: str, tensor: Tensor | None) -> None:
        if tensor is None:
            digest[name] = None
            return
        rows = tensor.detach().cpu().double().numpy().reshape(tensor.shape[0], -1).sum(axis=1)
        digest[name] = float((rows * np.arange(1, rows.size + 1)).sum())

    for name, tensor in zip(minibatch.obs._fields, minibatch.obs, strict=True):
        weighted(f"obs.{name}", tensor)
    for name in ("actions", "log_probs", "advantages", "returns", "values", "n_legal"):
        weighted(name, getattr(minibatch, name))
    digest["n_choice"] = int(minibatch.n_choice)
    digest["weight"] = float(minibatch.weight)
    return digest


class LearnerGroup(Peers):
    """Rank 0's end: the helpers, their queues, and the process group.

    Built by the coordinator once the training rectangle and the snapshot store exist, and
    closed with it. ``begin`` and ``end`` bracket each update.
    """

    def __init__(
        self,
        recipe: ReplicaRecipe,
        world: int,
        *,
        device: Any,
        champion: Callable[..., str | None] | None = None,
        printer: Callable[[str], None] | None = None,
    ) -> None:
        import torch.distributed as dist
        import torch.multiprocessing as mp

        super().__init__(0, world)
        if world < 2:
            raise ValueError("a learner group is for two GPUs or more")
        self.devices = learner_devices(device, world)
        self.backend = backend_for(device)
        self.champion = champion
        self._printer = printer
        self._procs: list[Any] = []
        self._inboxes: list[Any] = []
        self._outbox: Any = None
        self._joined = False
        self._store_dir = tempfile.mkdtemp(prefix="royalelearn-learners-")
        store_path = os.path.join(self._store_dir, "store")
        if self.backend == "nccl":
            # A collective that times out raises in the thread that waits on it, so a dead
            # helper stops the run with an error rather than a watchdog ending the process.
            os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
            os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "2")
        context = mp.get_context("spawn")
        self._outbox = context.Queue()
        try:
            for rank in range(1, world):
                inbox = context.Queue()
                process = context.Process(
                    target=_helper_main,
                    args=(
                        rank,
                        world,
                        self.backend,
                        store_path,
                        recipe,
                        self.devices[rank],
                        inbox,
                        self._outbox,
                        COLLECTIVE_TIMEOUT_S,
                    ),
                    name=f"royalelearn-learner-{rank}",
                    daemon=True,
                )
                process.start()
                self._inboxes.append(inbox)
                self._procs.append(process)
            self._await("built", STARTUP_TIMEOUT_S)
            dist.init_process_group(
                self.backend,
                store=dist.FileStore(store_path, world),
                rank=0,
                world_size=world,
                timeout=datetime.timedelta(seconds=COLLECTIVE_TIMEOUT_S),
            )
            self._joined = True
        except BaseException:
            self.close()
            raise
        #: The update's first batch, for the debug iteration's comparison.
        self._first_batch: Batch | None = None

    # -- each update -------------------------------------------------------

    def begin(
        self, update: PPOUpdate, buffer: RectBuffer, sched: Any, *, checking: bool
    ) -> None:
        """Hand the helpers this iteration and rank 0's weights."""
        self._check_alive()
        champion = self.champion
        payload = {
            "iteration": int(buffer.iteration),
            "cycles": int(buffer.cycles),
            "columns": buffer.replica_columns(),
            "sched": sched,
            "terms": [term.state() for term in update.extra_actor_terms],
            "champions": (champion(), champion(gated=True)) if champion else (None, None),
            "checking": bool(checking),
            "model": _model_layout(update.model),
        }
        for inbox in self._inboxes:
            inbox.put(("update", payload))
        self._first_batch = None
        self._guarded(self.broadcast_model, update.model)

    def note_first_batch(self, batch: Batch) -> None:
        """The update's first batch, whose helpers' first minibatches ``end`` checks."""
        self._first_batch = batch

    def reduce_gradients(self, params: Sequence[Parameter]) -> None:
        self._guarded(super().reduce_gradients, params)

    def broadcast_parameters(self, params: Sequence[Parameter]) -> None:
        self._guarded(super().broadcast_parameters, params)

    def end(self, update: PPOUpdate, buffer: RectBuffer, *, checking: bool) -> None:
        """Wait for every helper's update, and on a debug iteration compare its rows."""
        replies = self._await("done", FINISH_TIMEOUT_S)
        if not checking or self._first_batch is None:
            return
        batch = self._first_batch
        for rank, detail in sorted(replies.items()):
            chunks = batch.chunks(rank)
            if detail is None and not chunks:
                continue
            if detail is None or not chunks:
                raise AssertionError(
                    f"learner rank {rank} and rank 0 disagree on whether it had cells in the "
                    "update's first batch"
                )
            mine = minibatch_digest(buffer._gather(chunks[0], chunks[0].size / batch.n_samples))
            if mine != detail:
                differ = sorted(k for k in mine if mine.get(k) != detail.get(k))
                raise AssertionError(
                    f"learner rank {rank} trained on rows that differ from rank 0's gather of the "
                    f"same cells in {differ}: its view of the rectangle is not rank 0's"
                )

    # -- the processes -----------------------------------------------------

    def _guarded(self, collective: Callable[..., None], *args: Any) -> None:
        """A collective, and when it fails, what a helper said about why."""
        try:
            collective(*args)
        except Exception as exc:
            said = self._failure_message(wait=5.0)
            if said:
                raise HelperFailure(said) from exc
            raise

    def _failure_message(self, wait: float) -> str:
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                kind, rank, detail = self._outbox.get(timeout=0.2)
            except queue_module.Empty:
                continue
            if kind == "failed":
                return f"learner rank {rank} failed:\n{detail}"
        dead = self._dead()
        return f"learner ranks stopped: {dead}" if dead else ""

    def _dead(self) -> list[str]:
        return [
            f"rank {rank} exit code {process.exitcode}"
            for rank, process in enumerate(self._procs, start=1)
            if not process.is_alive()
        ]

    def _check_alive(self) -> None:
        dead = self._dead()
        if dead:
            said = self._failure_message(wait=1.0)
            raise HelperFailure(said or f"learner ranks stopped: {dead}")

    def _await(self, kind: str, timeout: float) -> dict[int, Any]:
        """One ``kind`` message from every helper, or the reason one will not come."""
        replies: dict[int, Any] = {}
        deadline = time.monotonic() + timeout
        while len(replies) < len(self._procs):
            try:
                got, rank, detail = self._outbox.get(timeout=0.5)
            except queue_module.Empty:
                if self._dead():
                    raise HelperFailure(f"learner ranks stopped: {self._dead()}") from None
                if time.monotonic() > deadline:
                    raise HelperFailure(
                        f"learner ranks {sorted(set(range(1, self.world)) - set(replies))} did "
                        f"not say {kind!r} within {timeout:.0f}s"
                    ) from None
                continue
            if got == "failed":
                raise HelperFailure(f"learner rank {rank} failed:\n{detail}")
            if got != kind:
                raise HelperFailure(f"learner rank {rank} said {got!r} where {kind!r} was due")
            replies[int(rank)] = detail
        return replies

    def close(self) -> None:
        """Stop the helpers and leave the process group. Safe to call twice."""
        import torch.distributed as dist

        for inbox in self._inboxes:
            with contextlib.suppress(Exception):
                inbox.put(("stop", None))
        deadline = time.monotonic() + STOP_TIMEOUT_S
        for process in self._procs:
            process.join(max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join(5.0)
        self._procs = []
        for channel in (*self._inboxes, self._outbox):
            if channel is not None:
                with contextlib.suppress(Exception):
                    channel.cancel_join_thread()
                    channel.close()
        self._inboxes = []
        self._outbox = None
        if self._joined:
            with contextlib.suppress(Exception):
                dist.destroy_process_group()
            self._joined = False
        shutil.rmtree(self._store_dir, ignore_errors=True)


def _model_layout(model: torch.nn.Module) -> list[tuple[str, tuple[int, ...], str]]:
    return [(k, tuple(v.shape), str(v.dtype)) for k, v in model.state_dict().items()]


def _helper_main(
    rank: int,
    world: int,
    backend: str,
    store_path: str,
    recipe: ReplicaRecipe,
    device: str,
    inbox: Any,
    outbox: Any,
    timeout_s: float,
) -> None:
    """A helper rank: build, join, then one update per message until told to stop."""
    import torch.distributed as dist

    replica: Replica | None = None
    joined = False
    try:
        if torch.device(device).type == "cuda":
            torch.cuda.set_device(torch.device(device))
        replica = recipe.build(rank=rank, world=world, device=torch.device(device))
        outbox.put(("built", rank, ""))
        dist.init_process_group(
            backend,
            store=dist.FileStore(store_path, world),
            rank=rank,
            world_size=world,
            timeout=datetime.timedelta(seconds=timeout_s),
        )
        joined = True
        peers = Peers(rank, world)
        replica.update.group = peers
        while True:
            kind, payload = inbox.get()
            if kind == "stop":
                break
            outbox.put(("done", rank, _replica_update(replica, payload)))
    except BaseException:
        with contextlib.suppress(Exception):
            outbox.put(("failed", rank, traceback.format_exc()))
    finally:
        if replica is not None:
            with contextlib.suppress(Exception):
                replica.close()
        if joined:
            with contextlib.suppress(Exception):
                dist.destroy_process_group()


def _replica_update(replica: Replica, payload: dict[str, Any]) -> dict[str, Any] | None:
    update = replica.update
    layout = _model_layout(update.model)
    if layout != payload["model"]:
        raise RuntimeError(
            "this helper's model is not rank 0's: the state dicts differ in "
            f"{sorted(set(map(str, layout)) ^ set(map(str, payload['model'])))[:4]}"
        )
    replica.buffer.take_replica_columns(payload["iteration"], payload["cycles"], payload["columns"])
    replica.set_champions(tuple(payload["champions"]))
    for term, state in zip(update.extra_actor_terms, payload["terms"], strict=True):
        # Loaded only when it differs: a term whose state is how far it has read a stream
        # restarts that stream on a load.
        if term.state() != state:
            term.load_state(state)
    return update.replica_step(replica.buffer, payload["sched"], checking=payload["checking"])


class RunReplica:
    """A helper rank's update and rectangle, built by ``RunRecipe``."""

    def __init__(self, buffer: RectBuffer) -> None:
        self.buffer = buffer
        self.update: PPOUpdate = None  # type: ignore[assignment]  # set once the terms exist
        self._champions: tuple[str | None, str | None] = (None, None)

    def champion(self, gated: bool = False) -> str | None:
        """``RunContext.champion`` on a helper: what rank 0 said with this iteration."""
        return self._champions[1 if gated else 0]

    def set_champions(self, champions: tuple[str | None, str | None]) -> None:
        self._champions = (champions[0], champions[1])

    def close(self) -> None:
        self.buffer.close()


class RunRecipe:
    """What the coordinator hands a helper rank: everything its update is built from that the
    helper cannot read off the config alone. Plain values, pickled once per helper.

    The config travels as builtins and is built again on the other side: a config with
    extension sections is an instance of a class made for those sections, which pickle cannot
    name. Its canonical JSON is checked to be rank 0's.
    """

    def __init__(self, *, config: Any, **fields: Any) -> None:
        from ..config import dump_config

        self.__dict__.update(fields)
        table = getattr(type(config), "_extensions", {})
        data = msgspec.to_builtins(config)
        self._config_sections = {name: data.pop(name, None) for name in table}
        self._config_data = data
        self._config_json = dump_config(config)

    @property
    def config(self) -> Any:
        """The run's config, built again from its builtins (once per process)."""
        built = self.__dict__.get("_config_built")
        if built is None:
            from ..config import dump_config
            from ..extensions import build_config

            built = build_config(dict(self._config_data), dict(self._config_sections))
            if dump_config(built) != self._config_json:
                raise RuntimeError(
                    "a helper rank built a config whose canonical JSON is not rank 0's: the two "
                    "processes would train different runs"
                )
            self.__dict__["_config_built"] = built
        return built

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_config_built", None)
        return state

    def build(self, *, rank: int, world: int, device: torch.device) -> RunReplica:
        from ..determinism import apply as apply_determinism
        from ..extensions import RunContext, active_extensions
        from ..ladder.snapshots import DiskSnapshotStore
        from ..rollout.inline import build_codec
        from ..seeding import TORCH_GLOBAL, derive_int, stream_path
        from .buffer import RectBuffer
        from .gae import GAE
        from .nets import DefaultNetworkFactory, build_actor
        from .ppo import PPOUpdate
        from .rows import RowCodec
        from .transform import attach_policy_transform

        config = self.config
        spec = self.spec
        apply_determinism(config.determinism.tier, torch_threads=config.determinism.torch_threads)
        torch.manual_seed(derive_int(config.master_seed, stream_path(TORCH_GLOBAL)) + rank)
        codec = build_codec(
            self.codec_path, spec, self.codec_table, tuple(config.extra_component_modules)
        )
        model = DefaultNetworkFactory(config.master_seed).build(spec, config.net, device=device)
        attach_policy_transform(model, config, spec)
        buffer = RectBuffer(
            spec,
            codec,
            run_id=self.run_id,
            cycles=self.cycles,
            n_slots=self.n_slots,
            device=device,
            discard_opponent_rows=config.ppo.discard_opponent_rows,
            segment_name=self.segment,
            attach=True,
        )
        buffer.set_static_planes(self.statics)

        def make_actor(where: Any) -> Any:
            return build_actor(spec, config.net).to(where)

        store = DiskSnapshotStore(
            Path(self.snapshot_root),
            template=self.snapshot_template,
            build=make_actor,
            max_resident=self.max_resident,
        )
        replica = RunReplica(buffer)
        context = RunContext(
            config=config,
            spec=spec,
            device=device,
            printer=lambda line: print(f"[learner {rank}] {line}", flush=True),
            model=model,
            build_actor=make_actor,
            artifact_spec=lambda: self.artifact_spec,
            row_codec=lambda: RowCodec(spec, codec, self.statics, device),
            resuming=self.resuming,
            ratio_atol=self.ratio_atol,
            precision=self.precision,
            policy_transform=getattr(model, "policy_transform", None),
            champion=replica.champion,
            load_snapshot=lambda snapshot_id, actor: store.load_into(snapshot_id, actor, device),
            learner_rank=rank,
            learner_world=world,
        )
        terms = [
            term
            for entry in active_extensions(config)
            for term in entry.extension.actor_terms(entry.section, context)
        ]
        replica.update = PPOUpdate(
            model,
            GAE(
                standardize_rewards=config.advantage.standardize_rewards,
                reward_clip=config.advantage.reward_clip,
            ),
            config.ppo,
            master_seed=config.master_seed,
            device=device,
            extra_actor_terms=terms,
            stage_layout=self.stage_layout,
        )
        return replica
