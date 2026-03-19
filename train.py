"""
Main training script: PPO + NovelD on Montezuma's Revenge.

Architecture:
  - Stable-Baselines3 PPO with CnnPolicy
  - NovelD intrinsic reward injected via a custom VecEnv wrapper

"""

import argparse
import os
from collections import deque
import numpy as np
import torch
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from noveld_reward import NovelDReward
import config


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO + NovelD on Montezuma's Revenge")

    # Training scale
    parser.add_argument("--timesteps", type=int,   default=config.TOTAL_TIMESTEPS, help="Total timesteps per env (total steps = timesteps × n_envs)")
    parser.add_argument("--n_envs",    type=int,   default=config.N_ENVS,          help="Number of parallel envs")

    # NovelD
    parser.add_argument("--alpha",     type=float, default=config.ALPHA,           help="NovelD alpha: novelty balance")
    parser.add_argument("--beta",      type=float, default=config.BETA,            help="NovelD beta: intrinsic reward scale")

    # PPO
    parser.add_argument("--lr",        type=float, default=config.PPO_LR,          help="PPO learning rate")
    parser.add_argument("--n_steps",   type=int,   default=config.PPO_N_STEPS,     help="PPO steps per update")
    parser.add_argument("--batch_size",type=int,   default=config.PPO_BATCH_SIZE,  help="PPO batch size")
    parser.add_argument("--ent_coef",  type=float, default=config.PPO_ENT_COEF,    help="PPO entropy coefficient")

    # RND
    parser.add_argument("--rnd_lr",    type=float, default=config.RND_LR,          help="RND predictor learning rate")
    parser.add_argument("--reward_norm_clip", type=float, default=None,            help="Clip upper bound for normalized r_int (default=5.0; 0=no clip)")

    # Misc
    parser.add_argument("--early_stop_window",    type=int,   default=0,     help="Stop if r_int is 0 for this many consecutive episodes (0=disabled)")
    parser.add_argument("--early_stop_threshold", type=float, default=0.001, help="Stop if mean r_int below this value")
    parser.add_argument("--save_name",            type=str,   default="noveld_montezuma", help="Model save filename")
    parser.add_argument("--wandb_project",type=str, default="noveld-montezuma", help="WandB project name")
    parser.add_argument("--wandb_name",  type=str,  default=None,               help="WandB run name (auto if not set)")
    parser.add_argument("--no_wandb",    action="store_true",                   help="Disable WandB logging")
    parser.add_argument("--r_pos",           type=float, default=0.0,    help="Reward per new grid cell visited this episode (0=disabled)")
    parser.add_argument("--novelty_reset",   action="store_true",        help="Reset episode novelty counter when external reward > 0 (re-incentivises exploration after key pickup)")
    parser.add_argument("--r_pos_global",    action="store_true",        help="Give r_pos only for cells never visited in training (stronger exploration signal)")
    parser.add_argument("--monitor",         action="store_true",        help="Show live training monitor window (requires opencv-python)")
    parser.add_argument("--full_episode",    action="store_true",        help="Treat all 5 lives as one episode (disable EpisodicLifeEnv)")
    parser.add_argument("--skull", type=str, default="normal",
                        choices=["normal", "freeze", "remove"],
                        help="Skull behaviour: normal / freeze (hold position) / remove (move off-screen)")

    return parser.parse_args()


# -----------------------------------------------------------------------
# VecEnv Wrapper: injects NovelD intrinsic reward
# -----------------------------------------------------------------------
_RAM_ROOM      = 3   # Montezuma RAM: current room number
_PLAYER_X_ADDR = 42  # RAM[42] = player X (estimated; adjust if grid looks wrong)
_PLAYER_Y_ADDR = 24  # RAM[24] = player Y (confirmed, inverted scale)
_CELL_W        = 8   # grid cell width  in game units
_CELL_H        = 8   # grid cell height in game units


class NovelDWrapper(VecEnvWrapper):
    """
    Wraps a VecEnv to add NovelD + position-grid intrinsic rewards.

    r_total = r_ext + r_int + r_pos
      r_int: NovelD intrinsic reward (RND-based novelty)
      r_pos: fixed bonus for entering a new grid cell this episode
    """

    def __init__(self, venv: VecEnv, device: torch.device, r_pos: float = 0.0,
                 novelty_reset: bool = False, r_pos_global: bool = False,
                 reward_norm_clip: float = None):
        super().__init__(venv)
        self.noveld         = NovelDReward(venv.num_envs, device, reward_norm_clip=reward_norm_clip)
        self._last_obs      = None
        self._r_pos         = r_pos
        self._novelty_reset = novelty_reset
        self._r_pos_global  = r_pos_global
        self._ep_cells  = [set() for _ in range(venv.num_envs)]  # reset each episode
        self._all_cells = [set() for _ in range(venv.num_envs)]  # lifetime

    def reset(self):
        obs = self.venv.reset()
        self._last_obs = obs
        self.noveld.reset()
        for s in self._ep_cells:
            s.clear()
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        # Compute intrinsic reward
        r_int = self.noveld.compute(self._last_obs, obs, dones)

        # Update RND predictor
        self.noveld.update_rnd(obs)

        # Grid-cell position reward
        r_pos = np.zeros(len(infos), dtype=np.float32)
        for i, info in enumerate(infos):
            info["r_ext"] = rewards[i]
            try:
                ram  = self.venv.envs[i].unwrapped.ale.getRAM()
                room = int(ram[_RAM_ROOM])
                cell = (room,
                        int(ram[_PLAYER_X_ADDR]) // _CELL_W,
                        int(ram[_PLAYER_Y_ADDR]) // _CELL_H)
                is_new = (cell not in self._all_cells[i]) if self._r_pos_global \
                         else (cell not in self._ep_cells[i])
                if is_new:
                    self._ep_cells[i].add(cell)
                    self._all_cells[i].add(cell)
                    r_pos[i] = self._r_pos
                info["room_number"] = room
            except AttributeError:
                pass
            info["r_int"] = r_int[i]
            info["r_pos"] = float(r_pos[i])

        # Reset novelty when the agent earns external reward (e.g. picks up key)
        # Clears the episode SimHash counter so all states become "first visit" again,
        # re-generating r_int and motivating the agent to keep exploring toward the door.
        if self._novelty_reset:
            for i, info in enumerate(infos):
                if info.get("r_ext", 0.0) > 0:
                    self.noveld.reset([i])
                    self._ep_cells[i].clear()

        # Reset per-episode cells on done
        for i, done in enumerate(dones):
            if done:
                self._ep_cells[i].clear()

        # Share total lifetime cell count via info for WandB logging
        total_cells = sum(len(s) for s in self._all_cells)
        for info in infos:
            info["total_cells"] = total_cells

        # Combined reward: r_ext + r_int + r_pos
        rewards = rewards + r_int + r_pos

        self._last_obs = obs
        return obs, rewards, dones, infos


# -----------------------------------------------------------------------
# Callback: log r_ext, r_int to WandB each episode
# -----------------------------------------------------------------------
class NovelDCallback(BaseCallback):
    def __init__(self, use_wandb=True, verbose=0,
                 early_stop_window=300, early_stop_threshold=0.0,
                 best_model_window=100, best_model_path="best_model"):
        super().__init__(verbose)
        self.use_wandb = use_wandb
        self._ep_r_ext  = None
        self._ep_r_int  = None
        self._ep_r_pos  = None
        self._ep_rooms  = None
        self._all_rooms = None

        # Early stopping: stop if mean r_int over last N episodes < threshold
        self.early_stop_window    = early_stop_window
        self.early_stop_threshold = early_stop_threshold
        self._recent_r_int        = deque(maxlen=early_stop_window or 1)

        # Best model: save when mean r_ext over last N episodes is highest
        self._best_model_window   = best_model_window
        self._best_model_path     = best_model_path
        self._recent_r_ext        = deque(maxlen=best_model_window)
        self._best_mean_r_ext     = -float("inf")

        # Periodic best model: save best within each 5M-step window
        self._window_size         = 5_000_000
        self._window_best_r_ext   = -float("inf")
        self._last_window_end     = 0

        # Periodic wandb log: upload every N steps even without episode end
        self._log_freq            = 10_000
        self._last_log_step       = 0
        self._recent_r_int_step   = deque(maxlen=50)
        self._recent_r_ext_step   = deque(maxlen=50)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        n_envs = len(infos)

        if self._ep_r_ext is None:
            self._ep_r_ext  = [0.0] * n_envs
            self._ep_r_int  = [0.0] * n_envs
            self._ep_r_pos  = [0.0] * n_envs
            self._ep_rooms  = [set() for _ in range(n_envs)]
            self._all_rooms = [set() for _ in range(n_envs)]

        for i, info in enumerate(infos):
            self._ep_r_ext[i] += info.get("r_ext", 0.0)
            self._ep_r_int[i] += info.get("r_int", 0.0)
            self._ep_r_pos[i] += info.get("r_pos", 0.0)

            # Track room number (set by NovelDWrapper via RAM read)
            room_id = info.get("room_number", None)

            if room_id is not None:
                self._ep_rooms[i].add(room_id)
                self._all_rooms[i].add(room_id)

            if "episode" in info:
                ep            = info["episode"]
                ep_rooms      = len(self._ep_rooms[i])
                total_rooms   = len(self._all_rooms[i])

                self._recent_r_ext_step.append(self._ep_r_ext[i])
                self._recent_r_int_step.append(self._ep_r_int[i])

                log = {
                    "episode/r_ext":      self._ep_r_ext[i],
                    "episode/r_int":      self._ep_r_int[i],
                    "episode/r_pos":      self._ep_r_pos[i],
                    "episode/r_total":    self._ep_r_ext[i] + self._ep_r_int[i] + self._ep_r_pos[i],
                    "episode/length":     ep["l"],
                    "episode/rooms":      ep_rooms,
                    "train/total_rooms":  total_rooms,
                    "train/total_cells":  info.get("total_cells", 0),
                    "train/timestep":     self.num_timesteps,
                }
                if self.use_wandb:
                    wandb.log(log)
                if self.verbose:
                    print(f"  [Episode] r_ext={self._ep_r_ext[i]:.1f}  "
                          f"r_int={self._ep_r_int[i]:.3f}  "
                          f"rooms={ep_rooms}  steps={ep['l']}")

                # Best model: save when rolling mean r_ext is highest
                self._recent_r_ext.append(self._ep_r_ext[i])
                if len(self._recent_r_ext) == self._best_model_window:
                    mean_r_ext = sum(self._recent_r_ext) / self._best_model_window
                    if mean_r_ext > self._best_mean_r_ext:
                        self._best_mean_r_ext = mean_r_ext
                        self.model.save(self._best_model_path)
                        if self.verbose:
                            print(f"  [BestModel] New best mean r_ext={mean_r_ext:.2f} "
                                  f"→ saved to {self._best_model_path}.zip")
                        if self.use_wandb:
                            wandb.log({"train/best_mean_r_ext": mean_r_ext},
                                      step=self.num_timesteps)

                # Periodic best model: best episode r_ext within each 5M-step window
                current_window = self.num_timesteps // self._window_size
                if current_window > self._last_window_end:
                    # New window started — reset tracker
                    self._window_best_r_ext = -float("inf")
                    self._last_window_end   = current_window
                if self._ep_r_ext[i] > self._window_best_r_ext:
                    self._window_best_r_ext = self._ep_r_ext[i]
                    window_label = current_window * 5
                    save_path = f"{self._best_model_path}_{window_label}M"
                    self.model.save(save_path)
                    if self.verbose:
                        print(f"  [WindowBest] r_ext={self._ep_r_ext[i]:.1f} "
                              f"→ saved to {save_path}.zip")

                # Early stopping check (disabled when window=0)
                if self.early_stop_window > 0:
                    self._recent_r_int.append(self._ep_r_int[i])
                    if (len(self._recent_r_int) == self.early_stop_window
                            and all(r < self.early_stop_threshold for r in self._recent_r_int)):
                        print(f"\n[EarlyStop] r_int has been 0 for all "
                              f"last {self.early_stop_window} episodes. "
                              f"Stopping training.")
                        return False  # signals SB3 to stop training

                self._ep_r_ext[i] = 0.0
                self._ep_r_int[i] = 0.0
                self._ep_r_pos[i] = 0.0
                self._ep_rooms[i] = set()

        # Periodic wandb log every _log_freq steps
        if (self.use_wandb
                and self.num_timesteps - self._last_log_step >= self._log_freq
                and len(self._recent_r_ext_step) > 0):
            wandb.log({
                "train/timestep":          self.num_timesteps,
                "train/mean_r_ext":        sum(self._recent_r_ext_step) / len(self._recent_r_ext_step),
                "train/mean_r_int":        sum(self._recent_r_int_step) / len(self._recent_r_int_step),
            })
            self._last_log_step = self.num_timesteps

        return True


# -----------------------------------------------------------------------
# Skull freeze wrapper
# -----------------------------------------------------------------------
# Known RAM addresses for the rolling skull in Montezuma's Revenge.
# If the skull still moves, run find_skull_addrs.py to identify the correct ones.

_SKULL_ADDRS = [47, 40]  # RAM[47] = skull X position, RAM[40] = skull velocity/direction


class FreezeSkullWrapper(gym.Wrapper):
    """
    Controls the rolling skull via RAM[47] (X position).

    remove=False : freeze skull at its starting position each episode.
    remove=True  : move skull permanently off-screen (X=0).
    """

    def __init__(self, env: gym.Env, remove: bool = False):
        super().__init__(env)
        self._remove = remove # if True, set skull RAM to 0 (off-screen); if False, freeze at starting position
        self._frozen: dict = {} # addr → value to write each frame (set on reset)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs) # on reset, record the RAM values to freeze or remove the skull
        if self._remove:
            self._frozen = {a: 0 for a in _SKULL_ADDRS}   # push off-screen
        else:
            ram = self.env.unwrapped.ale.getRAM() # read current RAM values to freeze skull in place
            self._frozen = {a: int(ram[a]) for a in _SKULL_ADDRS} # freeze at current position
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action) # each frame, write the frozen values back to RAM to control the skull
        ale = self.env.unwrapped.ale # access ALE to write directly to RAM and control the skull's behavior
        for addr, val in self._frozen.items():
            ale.setRAM(addr, val) # each frame, write the frozen values back to RAM to control the skull's behavior
        return obs, reward, terminated, truncated, info


# -----------------------------------------------------------------------
# Action space restriction wrapper
# -----------------------------------------------------------------------
_ALLOWED_ACTIONS = [0, 1, 2, 3, 4, 5, 10, 11, 12]
# NOOP(0), FIRE(1), UP(2), RIGHT(3), LEFT(4), DOWN(5),
# UPFIRE(10), RIGHTFIRE(11), LEFTFIRE(12)

class RestrictedActionWrapper(gym.ActionWrapper):
    """Maps a reduced action space [0, N) to a fixed subset of the full action space."""

    def __init__(self, env: gym.Env, allowed_actions: list):
        super().__init__(env)
        self._allowed = allowed_actions
        self.action_space = gym.spaces.Discrete(len(allowed_actions))

    def action(self, act):
        return self._allowed[act]


# -----------------------------------------------------------------------
# Build environment
# -----------------------------------------------------------------------
def make_env(n_envs: int, monitor: bool = False, skull: str = "normal", full_episode: bool = False):
    """Create vectorized + frame-stacked Atari env (5 lives, EpisodicLifeEnv).

    skull: "normal" | "freeze" | "remove"
    """

    def _init():
        render_mode = "rgb_array" if monitor else None
        env = gym.make(config.ENV_ID, render_mode=render_mode)
        if skull == "freeze":
            env = FreezeSkullWrapper(env, remove=False)
        elif skull == "remove":
            env = FreezeSkullWrapper(env, remove=True)
        env = RestrictedActionWrapper(env, _ALLOWED_ACTIONS)
        env = AtariWrapper(env, terminal_on_life_loss=not full_episode)  # EpisodicLifeEnv, NoopReset, MaxAndSkip, etc.
        return env

    env = make_vec_env(_init, n_envs=n_envs, seed=0)
    env = VecFrameStack(env, n_stack=4)
    return env


# -----------------------------------------------------------------------
# Callback: live monitor window showing all envs, highlights reward
# -----------------------------------------------------------------------
class LiveMonitorCallback(BaseCallback):
    """
    Displays a real-time grid of all training envs.
    Flashes a red border around any env that earns external reward.

    Requires make_env(monitor=True) and: pip install opencv-python
    """

    def __init__(self, n_envs: int, display_freq: int = 8, verbose: int = 0):
        super().__init__(verbose)
        self.n_envs       = n_envs
        self.display_freq = display_freq  # refresh every N _on_step calls
        self._tick        = 0

    def _on_step(self) -> bool:
        self._tick += 1
        if self._tick % self.display_freq != 0:
            return True

        try:
            import cv2
        except ImportError:
            print("  [Monitor] Install: pip install opencv-python")
            return True

        frames = self.training_env.env_method("render")
        frames = [f for f in frames if isinstance(f, np.ndarray)]
        if not frames:
            return True

        # Build grid: 4 columns
        n_cols = min(4, len(frames))
        n_rows = (len(frames) + n_cols - 1) // n_cols
        # Pad to full grid
        blank = np.zeros_like(frames[0])
        frames += [blank] * (n_rows * n_cols - len(frames))

        # Red border on envs that got reward this step
        infos = self.locals.get("infos", [])
        for i, info in enumerate(infos):
            if info.get("r_ext", 0.0) > 0:
                h, w = frames[i].shape[:2]
                cv2.rectangle(frames[i], (0, 0), (w - 1, h - 1), (255, 0, 0), 6)

        rows = [np.concatenate(frames[r * n_cols:(r + 1) * n_cols], axis=1)
                for r in range(n_rows)]
        grid = np.concatenate(rows, axis=0)

        cv2.imshow("Training Monitor", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        return True

    def _on_training_end(self):
        try:
            import cv2
            cv2.destroyAllWindows()
        except ImportError:
            pass



def main():
    args = parse_args()

    # Print active config
    print("=" * 50)
    print("NovelD Training Config")
    print("=" * 50)
    print(f"  timesteps : {args.timesteps:,}  (total={args.timesteps * args.n_envs:,})")
    print(f"  n_envs    : {args.n_envs}")
    print(f"  alpha     : {args.alpha}")
    print(f"  beta      : {args.beta}")
    print(f"  lr        : {args.lr}")
    print(f"  rnd_lr    : {args.rnd_lr}")
    print(f"  ent_coef  : {args.ent_coef}")
    print(f"  save_name : {args.save_name}")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Override config with args
    config.ALPHA  = args.alpha
    config.BETA   = args.beta
    config.RND_LR = args.rnd_lr

    # Init WandB
    use_wandb = not args.no_wandb
    if use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),   # logs all hyperparams automatically
            sync_tensorboard=False,
        )
        # Use actual training timestep as x-axis for all custom metrics
        wandb.define_metric("train/timestep")
        wandb.define_metric("episode/*", step_metric="train/timestep")
        wandb.define_metric("train/total_rooms", step_metric="train/timestep")
        wandb.define_metric("train/total_cells", step_metric="train/timestep")
        wandb.define_metric("train/best_mean_r_ext", step_metric="train/timestep")
        print(f"WandB run: {run.url}\n")

    # Build env
    env = make_env(args.n_envs, monitor=args.monitor, skull=args.skull, full_episode=args.full_episode)
    clip_val = np.inf if (args.reward_norm_clip is not None and args.reward_norm_clip == 0) \
               else args.reward_norm_clip
    env = NovelDWrapper(env, device, r_pos=args.r_pos,
                        novelty_reset=args.novelty_reset,
                        r_pos_global=args.r_pos_global,
                        reward_norm_clip=clip_val)

    # PPO
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=config.PPO_N_EPOCHS,
        gamma=config.PPO_GAMMA,
        gae_lambda=config.PPO_GAE_LAMBDA,
        clip_range=config.PPO_CLIP_RANGE,
        ent_coef=args.ent_coef,
        verbose=1,
        device=device,
    )

    os.makedirs("checkpoints", exist_ok=True)
    callbacks = [
        NovelDCallback(use_wandb=use_wandb, verbose=1,
                       early_stop_window=args.early_stop_window,
                       early_stop_threshold=args.early_stop_threshold,
                       best_model_window=100,
                       best_model_path="best_model"),
        CheckpointCallback(save_freq=200_000, save_path="checkpoints/",
                           name_prefix=args.save_name, verbose=1),
    ]

    # Optional: live monitor window
    if args.monitor:
        callbacks.append(LiveMonitorCallback(n_envs=args.n_envs, verbose=1))

    if use_wandb:
        model_save_path = f"models/{wandb.run.id}"
        os.makedirs(model_save_path, exist_ok=True)
        callbacks.append(WandbCallback(
            gradient_save_freq=1000,
            model_save_path=model_save_path,
            verbose=0,
        ))

    print("Starting training...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(args.save_name)
    print(f"Model saved to {args.save_name}.zip")

    if use_wandb:
        wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
