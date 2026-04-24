import numpy as np


class LinearThompsonSampling:
    def __init__(
        self,
        n_actions: int,
        n_features: int,
        lambda_reg: float = 1.0,
        sigma: float = 0.1,
        seed: int | None = None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lambda_reg = lambda_reg
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

        self.A = np.array([
            lambda_reg * np.eye(n_features)
            for _ in range(n_actions)
        ])

        self.b = np.zeros((n_actions, n_features))

    def select_action(self, context: np.ndarray) -> int:
        context = np.asarray(context, dtype=np.float64)

        sampled_rewards = np.zeros(self.n_actions)

        for action in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[action])
            theta_hat = A_inv @ self.b[action]

            cov = (self.sigma ** 2) * A_inv

            theta_sample = self.rng.multivariate_normal(
                mean=theta_hat,
                cov=cov,
            )

            sampled_rewards[action] = context @ theta_sample

        return int(np.argmax(sampled_rewards))

    def update(
        self,
        context: np.ndarray,
        action: int,
        reward: float,
    ) -> None:
        context = np.asarray(context, dtype=np.float64)

        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context


# -------------------------
# Smoke test
# -------------------------

if __name__ == "__main__":
    print("Running LinearThompsonSampling smoke test...")

    from src.cbp.data.base_dataset import MovieLensDataset
    from src.cbp.envs.bandit_env import BanditEnv

    ds = MovieLensDataset(data_dir="data/raw/ml-100k")
    data = ds.load_bandit_data()

    X = data["X"]
    actions = data["actions"]
    rewards = data["rewards"]

    unique_actions = np.unique(actions)
    idx_to_action = {i: a for i, a in enumerate(unique_actions)}

    n_actions = len(unique_actions)
    n_features = X.shape[1]

    env = BanditEnv(X, actions, rewards, mode="full")

    policy = LinearThompsonSampling(
        n_actions=n_actions,
        n_features=n_features,
        lambda_reg=1.0,
        sigma=0.1,
        seed=42,
    )

    env.reset()

    total_reward = 0
    steps = 0

    for _ in range(50):
        context = env.get_context()

        action_idx = policy.select_action(context)
        action = idx_to_action[action_idx]

        reward, done, info = env.step(action)

        assert 0 <= action_idx < n_actions
        assert reward in [0, 1]

        policy.update(context, action_idx, reward)

        total_reward += reward
        steps += 1

        print(
            f"Step {steps} | "
            f"action_idx={action_idx} | "
            f"action={action} | "
            f"reward={reward}"
        )

        if done:
            break

    print(f"Total reward: {total_reward}")
    print(f"Avg reward: {total_reward / steps:.4f}")
    print("LinearThompsonSampling smoke test passed.")