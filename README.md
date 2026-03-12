# NovelD on Montezuma's Revenge

PPO + NovelD intrinsic motivation for exploration in Montezuma's Revenge (Atari), built on [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

## Method

[NovelD](https://arxiv.org/abs/2102.09300) extends RND (Random Network Distillation) by rewarding the agent for moving *toward* novel states rather than simply being in them, and by applying an episode-level first-visit mask to prevent the agent from repeatedly exploiting the same state:

$$r^i_t = \max\left(\text{novel}(s_{t+1}) - \alpha \cdot \text{novel}(s_t),\ 0\right) \cdot \mathbf{1}\left[N_e(s_{t+1}) = 1\right]$$

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+. A CUDA-capable GPU is recommended.

## Training

```bash
python train.py \
  --skull freeze \
  --timesteps 500000000 \
  --novelty_reset \
  --n_envs 10 \
  --ent_coef 0.05 \
  --rnd_lr 1e-5 \
  --n_steps 512 \
  --batch_size 512
```

Add `--monitor` to display a live grid of all training environments.
Training metrics are logged to [Weights & Biases](https://wandb.ai) by default. Use `--no_wandb` to disable.

Key arguments:

- `--skull freeze` — freeze the rolling skull in room 1 via RAM writes so the agent can safely reach the key
- `--novelty_reset` — reset the episode visit counter $N_e$ whenever the agent receives extrinsic reward, restoring the intrinsic exploration signal after key pickup
- `--n_envs` — number of parallel environments (default: 8)
- `--alpha` — NovelD novelty balance coefficient (default: 0.5)
- `--beta` — intrinsic reward scale applied to the normalized novelty signal (default: 0.05)
- `--rnd_lr` — learning rate for the RND predictor network (default: 1e-4)

## Evaluation

```bash
python render.py --model noveld_montezuma.zip --skull freeze --episodes 5
```

## Project Structure

```
train.py            # Main training script (PPO + NovelD wrapper)
render.py           # Load and visualise a trained model
noveld_reward.py    # NovelD intrinsic reward computation
rnd_network.py      # RND target and predictor networks
episode_counter.py  # SimHash-based episode visit counter
config.py           # Default hyperparameters
requirements.txt    # Python dependencies
```
