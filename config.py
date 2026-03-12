"""
Hyperparameters for NovelD + PPO on Montezuma's Revenge
"""

# Environment
ENV_ID = "ALE/MontezumaRevenge-v5"
N_ENVS = 8                   # parallel environments
TOTAL_TIMESTEPS = 10_000_000

# PPO
PPO_LR = 2.5e-4
PPO_N_STEPS = 128
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.1
PPO_ENT_COEF = 0.01

# RND
RND_OUTPUT_DIM = 512          # output dimension of target/predictor
RND_LR = 1e-4                 # predictor optimizer learning rate

# NovelD
ALPHA = 0.5                   # novelty balance: novel(s_{t+1}) - alpha * novel(s_t)
BETA = 0.05                   # intrinsic reward scale: r_total = r_ext + beta * r_int

# Normalization
OBS_NORM_CLIP = 5.0           # clip value for observation normalization
REWARD_NORM_CLIP = 5.0        # clip value for reward normalization
