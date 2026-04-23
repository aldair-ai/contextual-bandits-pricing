import numpy as np

from src.cbp.data.base_dataset import MovieLensDataset
from src.cbp.envs.bandit_env import BanditEnv
from src.cbp.policies.epsilon_greedy import EpsilonGreedy


def run_single_policy(X, actions, rewards, epsilon=0.1, max_steps=1000):
    # map actions → indices
    unique_actions = np.unique(actions)
    action_to_idx = {a: i for i, a in enumerate(unique_actions)}
    idx_to_action = {i: a for i, a in enumerate(unique_actions)}

    # init env + policy
    env = BanditEnv(X, actions, rewards, mode="full")
    policy = EpsilonGreedy(n_actions=len(unique_actions), epsilon=epsilon)

    env.reset()

    total_reward = 0
    rewards_history = []

    for t in range(max_steps):
        context = env.get_context()

        action_idx = policy.select_action()
        action = idx_to_action[action_idx]

        reward, done, _ = env.step(action)

        policy.update(action_idx, reward)

        total_reward += reward
        rewards_history.append(reward)

        if done:
            break

    return total_reward, rewards_history


# -------------------------
# Smoke test / minimal run
# -------------------------
if __name__ == "__main__":
    print("Running minimal experiment...")

    # load dataset
    ds = MovieLensDataset(data_dir="data/raw/ml-100k")
    data = ds.load_bandit_data()

    X = data["X"]
    actions = data["actions"]
    rewards = data["rewards"]

    # run single policy
    total_reward, rewards_history = run_single_policy(
        X,
        actions,
        rewards,
        epsilon=0.1,
        max_steps=1000,
    )

    # basic checks
    assert len(rewards_history) > 0
    assert all(r in [0, 1] for r in rewards_history)

    print(f"Total reward: {total_reward}")
    print(f"Avg reward: {np.mean(rewards_history):.4f}")
    print("Run successful.")