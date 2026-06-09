import os
import sys


def _find_root():
    candidates = [
        os.getcwd(),
        "/kaggle_simulations/agent",
        "/kaggle/working",
    ]
    for root in candidates:
        if os.path.exists(os.path.join(root, "src")):
            return root
    return os.getcwd()


_ROOT = _find_root()
_SRC = os.path.join(_ROOT, "src")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

os.environ.setdefault(
    "ORBIT_PPO_MISSION_MODEL",
    os.path.join(_ROOT, "checkpoints", "ppo_selfplay_best.pt"),
)

try:
    from orbit_rl.inference_agent import agent as _agent
except Exception:
    _agent = None


def agent(obs, config=None):
    if _agent is not None:
        try:
            return _agent(obs, config)
        except Exception:
            pass
    return []
