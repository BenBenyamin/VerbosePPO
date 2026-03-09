import gymnasium as gym

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from ppo import PPO

# Define the env
env = gym.make("CartPole-v1")
device = "cuda"

# Init the model
model = PPO(
    env=env,
    device=device,
    num_steps=1024,
    num_minibatches=4,
    update_epochs=4,
    learning_rate=2.5e-4,
    anneal_lr=True,
    norm_adv=True,
    clip_coef=0.2,
    vf_coef=0.5,
    ent_coef=0.0,
    kl_coeff=0.02,
    max_grad_norm=0.5,
    target_kl=None,
    hidden_size=64,
    init_weights=True,
)

model.load(Path(__file__).parent / "PPO_100_000_CartPoleV1")
model.learn(100_000)
model.save("PPO_200_000_CartPoleV1")

returns = []
for ep in range(100):
    obs, _ = env.reset()
    done = False
    ep_ret = 0.0

    while not done:
        action  = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        ep_ret += float(reward)

    returns.append(ep_ret)

env.close()

print(f"Evaluation over 100 episodes:")
print(f"  returns: {returns}, mean: {sum(returns)/len(returns)}")