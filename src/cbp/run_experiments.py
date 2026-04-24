import numpy as np

from src.cbp.data.base_dataset import MovieLensDataset
from src.cbp.envs.bandit_env import BanditEnv
from src.cbp.policies.epsilon_greedy import EpsilonGreedy
from src.cbp.policies.linear_thompson import LinearThompsonSampling


def run_single_policy(X, actions, rewards, policy, max_steps=1000):
    unique_actions = np.unique(actions)
    idx_to_action = {i: a for i, a in enumerate(unique_actions)}

    env = BanditEnv(X, actions, rewards, mode="full")
    env.reset()

    total_reward = 0
    rewards_history = []

    for _ in range(max_steps):
        context = env.get_context()

        action_idx = policy.select_action(context)
        action = idx_to_action[action_idx]

        reward, done, _ = env.step(action)

        policy.update(context, action_idx, reward)

        total_reward += reward
        rewards_history.append(reward)

        if done:
            break

    return total_reward, rewards_history


def run_experiments(X, actions, rewards, max_steps=1000):
    unique_actions = np.unique(actions)

    n_actions = len(unique_actions)
    n_features = X.shape[1]

    policies = {
        "EpsilonGreedy": EpsilonGreedy(
            n_actions=n_actions,
            epsilon=0.1,
            seed=42,
        ),
        "LinearThompsonSampling": LinearThompsonSampling(
            n_actions=n_actions,
            n_features=n_features,
            lambda_reg=1.0,
            sigma=0.1,
            seed=42,
        ),
    }

    results = {}

    for policy_name, policy in policies.items():
        total_reward, rewards_history = run_single_policy(
            X=X,
            actions=actions,
            rewards=rewards,
            policy=policy,
            max_steps=max_steps,
        )

        results[policy_name] = {
            "total_reward": total_reward,
            "avg_reward": float(np.mean(rewards_history)),
            "rewards_history": rewards_history,
        }

    return results


if __name__ == "__main__":
    print("Running minimal experiment...")

    ds = MovieLensDataset(data_dir="data/raw/ml-100k")
    data = ds.load_bandit_data()

    X = data["X"]
    actions = data["actions"]
    rewards = data["rewards"]

    results = run_experiments(
        X=X,
        actions=actions,
        rewards=rewards,
        max_steps=1000,
    )

    for policy_name, metrics in results.items():
        rewards_history = metrics["rewards_history"]

        assert len(rewards_history) > 0
        assert all(r in [0, 1] for r in rewards_history)

        print("\n" + "=" * 40)
        print(policy_name)
        print("=" * 40)
        print(f"Total reward: {metrics['total_reward']}")
        print(f"Avg reward: {metrics['avg_reward']:.4f}")

    print("\nRun successful.")