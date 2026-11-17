"""Score a microduck checkpoint headlessly and write the metrics as JSON."""
import pathlib
import sys

_RL_AGENT = str(pathlib.Path(__file__).resolve().parents[1])
if _RL_AGENT not in sys.path:
    sys.path.insert(0, _RL_AGENT)

from mjlab_core.evaluate import evaluate, main  # noqa: E402,F401

if __name__ == "__main__":
    main()
