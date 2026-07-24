# Proposal: switch `ParallelPyEnvironment` workers from `fork` to `spawn`

**Status:** proposed, not implemented. Written 2026-07-22 after a fork+CUDA
segfault killed TRAIN job `6a601d44a8350bdc89d19a60` at 70% (train_step
177057/250000), ~14 hours into the run. See `scripts/watchdog.log` and this
job's `eval_error` history for the incident record. Recovered via manual
checkpoint restore + (as of the same day) `Watchdog.ps1`'s new FAILED-job
auto-resume path (option "C" from that incident's follow-up) rather than
this fix, which was deferred as higher-risk/higher-effort.

## Root cause recap

`tf_agents.environments.parallel_py_environment.ProcessPyEnvironment.start()`
creates each parallel env worker via:

```python
mp_context = multiprocessing.get_context()
self._process = mp_context.Process(target=self._worker, args=(conn,))
```

`multiprocessing.get_context()` with no explicit method name resolves to the
platform default, which on Linux is **`fork`**. `robotaxi.py` never calls
`multiprocessing.set_start_method(...)`, so it inherits this default.

Separately, `robotaxi.py` initializes TensorFlow's CUDA/GPU runtime at
**module import time** (the `[gpu] enabled memory growth on 1 GPU(s)` log
line, which appears before "Polling for jobs..." - i.e. before any job, and
therefore before any env-worker fork, has even been picked up).

Forking a process after CUDA/TF has initialized native threads, mutexes, and
GPU driver state is a well-documented unsafe pattern (see e.g. the CUDA
programming guide's warning against `fork()` after context creation, and
numerous TensorFlow issues describing exactly this symptom). The child
process gets a byte-for-byte memory copy of the parent, but **not** copies of
the parent's other live OS threads or the true state of any native mutex that
happened to be held at fork time - so any native C++ state that assumed a
consistent multi-threaded view can be silently corrupted in the child from
the moment of fork onward.

In this incident, the corruption didn't manifest immediately - the worker
ran correctly, driving real Unity episodes, for hours. It finally crashed
when Python's garbage collector reaped a leftover `ConcreteFunction` object
(inherited from the parent's memory at fork time) inside the worker's
background asyncio thread (`envs.py:188`), and its `__del__` /
`weakref.finalize` callback called into TensorFlow's C++
`context.py:remove_function` - hitting the corrupted state and segfaulting.

This is inherently **sporadic**: it depends on exactly when GC happens to
collect that particular object relative to whatever native state was
disturbed at fork time. It is not tied to any of the recent demo-buffer /
replay-buffer changes.

## Proposed fix

Force the `spawn` start method for every `multiprocessing.Process` created
via `ParallelPyEnvironment`, so env workers are always freshly-started
Python interpreters (re-importing everything cold) rather than forked
copies of a CUDA-touched parent. This eliminates the unsafe-fork-after-CUDA
pattern entirely - it's the same fix commonly recommended in TF+multiprocessing
GitHub issues for this exact crash signature.

```python
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)  # must run before ANY
                                                        # Process/Pool is created,
                                                        # ideally at the very top
                                                        # of robotaxi.py before the
                                                        # TF/CUDA-touching imports.
```

## Why this is a bigger/riskier change than it looks

`spawn` does **not** fork - it starts a brand-new Python interpreter and
re-imports the target module, then **pickles** the `Process`'s `target` and
`args` to hand to that new interpreter. This has real implications here:

1. **Env constructors must be picklable.** `ParallelPyEnvironment` is
   constructed with a list of zero-arg callables (`env_constructors`) in
   `build_train_env()` (`robotaxi.py`). If any of these are `lambda`s or
   closures over local variables (need to verify - likely candidates given
   `make_env(...)` is called with per-actor `grpc_addr`/`actor_index`
   arguments), they are **not** picklable and `spawn` will fail immediately
   with a `PicklingError`. These would need to become top-level, module-level
   functions (e.g. via `functools.partial(make_env, grpc_addr=..., actor_index=...)`,
   which IS picklable, since `functools.partial` pickles its target function
   by reference plus its plain-data args).
2. **Every module-level side effect at import time re-runs in each worker.**
   With `fork`, a worker inherits already-completed import-time work (GPU
   memory-growth config, uvloop policy install, git-provenance printing,
   etc.) for free. With `spawn`, each worker re-executes `robotaxi.py`'s
   top-level code from scratch on import - including re-attempting GPU
   memory-growth configuration and potentially re-acquiring the trainer
   singleton lock file if that logic isn't guarded by `if __name__ ==
   '__main__':`. This needs a careful audit of everything at module scope
   before `do_job`/`main` to ensure it's either idempotent or properly
   guarded against re-running in workers.
3. **Slower worker startup.** Cold-starting Python + re-importing
   TensorFlow, tf-agents, grpc, etc. per worker takes a few seconds instead
   of the near-instant `fork`. Given `env_spawn` currently costs ~0.1s and
   jobs run for hours, this is very likely an acceptable one-time cost per
   job - but should be measured.
4. **Windows-container parity is irrelevant here** (this runs in the Linux
   `sim-controller` container) but worth noting `spawn` is Windows' only
   option, so this code path is at least well-trodden in the wild.

## Recommended validation before rollout

1. Grep `build_train_env` (and anywhere else `ParallelPyEnvironment` or
   `mp_context.Process`/`multiprocessing.Process` is constructed) for the
   actual `env_constructors` callables passed in, and confirm/convert them
   to picklable top-level callables (e.g. `functools.partial`).
2. Audit `robotaxi.py` module-level code (everything that runs on import,
   before `if __name__ == '__main__':`/`multiprocessing.handle_main(...)`)
   for anything unsafe to re-run in a spawned worker (file locks, one-time
   log lines that would now spam per-worker, etc.).
3. Add `multiprocessing.set_start_method('spawn', force=True)` at the very
   top of `robotaxi.py`, before the TensorFlow import if at all possible
   (spawn's benefit is largest when the *worker's own* GPU/CUDA init hasn't
   happened yet at fork-equivalent time - though for `spawn` there is no
   fork-time to worry about at all, so this is more about keeping the
   parent's own pre-fork CUDA touch from ever being copied into a child in
   the first place).
4. Run a short (~30-60 min) validation job with 2 gyms to confirm workers
   start correctly, drive normally, and no new pickling/import-order errors
   appear, before trusting it on a multi-day run.
5. Keep the `Watchdog.ps1` FAILED-job auto-resume path either way - even
   with `spawn`, worker crashes of other kinds (Unity itself crashing,
   OOM, etc.) can still happen, and auto-resume-from-checkpoint is cheap
   insurance regardless of this fix's fate.
