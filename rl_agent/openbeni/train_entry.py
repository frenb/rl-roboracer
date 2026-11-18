"""Train entry that installs cfg hooks without modifying third_party/open-beni.

``mjlab_core.launch`` invokes this as ``python train_entry.py <task> ...``.
Hooks mirror microduck_rl's ``MICRODUCK_*_HOOK`` pattern so reward and
experiment designs can patch env/agent cfgs before ``beni_rl.train`` builds them.
"""
from __future__ import annotations

import importlib.util
import os
import sys

CFG_HOOK_ENV = "OPENBENI_CFG_HOOK"
RL_CFG_HOOK_ENV = "OPENBENI_RL_CFG_HOOK"


def _load_hook(spec: str):
    path, _, name = spec.partition(":")
    if not path or not name:
        raise SystemExit(f"cfg hook must be path.py:function, got {spec!r}")
    mod_name = f"_openbeni_hook_{abs(hash(path))}"
    spec_obj = importlib.util.spec_from_file_location(mod_name, path)
    if spec_obj is None or spec_obj.loader is None:
        raise SystemExit(f"cannot load cfg hook module {path!r}")
    mod = importlib.util.module_from_spec(spec_obj)
    sys.modules[mod_name] = mod
    spec_obj.loader.exec_module(mod)
    fn = getattr(mod, name, None)
    if not callable(fn):
        raise SystemExit(f"cfg hook {spec!r}: {name!r} is not callable")
    return fn


def install_hooks() -> None:
    """Wrap ``load_env_cfg`` / ``load_rl_cfg`` when the corresponding env vars are set."""
    env_hook = os.environ.get(CFG_HOOK_ENV)
    rl_hook = os.environ.get(RL_CFG_HOOK_ENV)
    if not env_hook and not rl_hook:
        return

    import mjlab.tasks.registry as registry

    env_fn = _load_hook(env_hook) if env_hook else None
    rl_fn = _load_hook(rl_hook) if rl_hook else None
    orig_env = registry.load_env_cfg
    orig_rl = registry.load_rl_cfg

    def load_env_cfg_wrapped(task_id, *args, **kwargs):
        cfg = orig_env(task_id, *args, **kwargs)
        if env_fn is not None:
            play = bool(kwargs.get("play", False))
            if args and isinstance(args[0], bool):
                play = args[0]
            env_fn(task_id, cfg, play)
        return cfg

    def load_rl_cfg_wrapped(task_id, *args, **kwargs):
        cfg = orig_rl(task_id, *args, **kwargs)
        if rl_fn is not None:
            rl_fn(task_id, cfg)
        return cfg

    if env_fn:
        registry.load_env_cfg = load_env_cfg_wrapped
    if rl_fn:
        registry.load_rl_cfg = load_rl_cfg_wrapped

    # beni_rl.train binds names at import time; keep both in sync.
    try:
        import beni_rl.train as beni_train
        if env_fn:
            beni_train.load_env_cfg = registry.load_env_cfg
        if rl_fn:
            beni_train.load_rl_cfg = registry.load_rl_cfg
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    install_hooks()
    from beni_rl.train import main as beni_main
    beni_main()


if __name__ == "__main__":
    main()
