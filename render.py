"""
Render a trained NovelD model playing Montezuma's Revenge.

Usage:
  python render.py                              # loads noveld_montezuma.zip
  python render.py --model my_model.zip
  python render.py --model my_model.zip --episodes 5
  python render.py --model my_model.zip --skull freeze
"""

import argparse
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper

import config
from train import FreezeSkullWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Render trained NovelD agent")
    parser.add_argument("--model",    type=str, default="noveld_montezuma.zip", help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=3,                      help="Number of episodes to render")
    parser.add_argument("--skull",    type=str, default="normal",
                        choices=["normal", "freeze", "remove"],
                        help="Skull behaviour: normal / freeze / remove")
    return parser.parse_args()


def main():
    args = parse_args()

    def _init():
        env = gym.make(config.ENV_ID, render_mode="human")
        if args.skull == "freeze":
            env = FreezeSkullWrapper(env, remove=False)
        elif args.skull == "remove":
            env = FreezeSkullWrapper(env, remove=True)
        env = AtariWrapper(env)
        return env

    env = make_vec_env(_init, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load(args.model, env=env)
    print(f"Loaded model: {args.model}")

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
            steps += 1
            done = done[0]

        print(f"Episode {ep+1}: reward={total_reward:.1f}  steps={steps}")

    env.close()


if __name__ == "__main__":
    main()
