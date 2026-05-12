import os
import argparse
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from env import SocialDemocracyEnv
from memory_profiler import memory_usage
import psutil

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--headless",   action="store_true",  default=True,
                    help="run browser headless (default: on)")
parser.add_argument("--no-headless", dest="headless", action="store_false",
                    help="show the browser window")
parser.add_argument("--debug",      action="store_true",  default=False,
                    help="print step-by-step game state")
parser.add_argument("--timesteps",  type=int, default=500_000)
parser.add_argument("--load",       type=str, default=None,
                    help="path to a saved .zip to resume training from")
args = parser.parse_args()

# ── env ───────────────────────────────────────────────────────────────────────
# DummyVecEnv only — SubprocVecEnv forks the process and Playwright's semaphores
# do not survive a fork, which is what caused your segfault.
# Single-env training is fine here because the bottleneck is the browser, not the CPU.
def make():
    env = SocialDemocracyEnv(headless=args.headless, debug=args.debug)
    return ActionMasker(env, lambda e: e.action_masks())

def log_memory():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # in MB
    print(f"[memory] {mem:.2f} MB", flush=True)
print(f"device : {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"headless: {args.headless}  debug: {args.debug}  timesteps: {args.timesteps:,}")

vec_env = DummyVecEnv([make])
n_steps = 32  # env steps per PPO update — keep it small to avoid OOM, since the browser is the bottleneck, not the CPU
total = args.timesteps
total = ((total + n_steps - 1) // n_steps) * n_steps  # round down to nearest n_steps

# ── model ─────────────────────────────────────────────────────────────────────
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs",        exist_ok=True)

class MemoryCallback(BaseCallback):
    def _on_step(self):
        if self.n_calls % 100 == 0:  # log every 100 steps
            proc = __import__("psutil").Process(__import__("os").getpid())
            mem = proc.memory_info().rss / (1024 * 1024)  # in MB
            print(f"[memory] {mem:.2f} MB", flush=True)
        return True
if args.load:
    print(f"resuming from {args.load}")
    model = MaskablePPO.load(args.load, env=vec_env, tensorboard_log="logs/")
else:
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="logs/",
        policy_kwargs=dict(net_arch=[256, 256]),
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,      # high gamma — rewards come very late in long episodes
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,    # keeps exploration alive
        learning_rate=3e-4,
    )

# ── train ─────────────────────────────────────────────────────────────────────
checkpoint_cb = CheckpointCallback(
    save_freq=10_000,
    save_path="checkpoints/",
    name_prefix="spd",
    verbose=1,
)

print("\nrun:  tensorboard --logdir logs/\n")

try:
    model.learn(
        total_timesteps=total,
        callback=[checkpoint_cb, MemoryCallback()],
        progress_bar=True,
        reset_num_timesteps=args.load is None,
    )
    print("training complete")
except KeyboardInterrupt:
    print("interrupted")
finally:
    model.save("spd_final")
    print("saved -> spd_final.zip")
    vec_env.close()