import numpy as np


class EpsilonGreedy:
    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        self.counts = np.zeros(n_actions, dtype=np.int32)
        self.values = np.zeros(n_actions, dtype=np.float32)

    def select_action(self, context: np.ndarray | None = None) -> int:
        """
        Select an action using epsilon-greedy.

        context is accepted for API compatibility with contextual algorithms
        like LinUCB, but vanilla epsilon-greedy does not use it.
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        return int(np.argmax(self.values))

    def update(
        self,
        context: np.ndarray | None,
        action: int,
        reward: float,
    ) -> None:
        """
        Update the estimated value of the selected action.

        context is accepted for API compatibility with LinUCB.
        """
        self.counts[action] += 1
        n = self.counts[action]

        self.values[action] += (reward - self.values[action]) / n


# -------------------------
# Smoke test
# -------------------------

if __name__ == "__main__":
    print("Running EpsilonGreedy smoke test...")

    from src.cbp.data.base_dataset import MovieLensDataset
    from src.cbp.envs.bandit_env import BanditEnv

    ds = MovieLensDataset(data_dir="data/raw/ml-100k")
    data = ds.load_bandit_data()

    X = data["X"]
    actions = data["actions"]
    rewards = data["rewards"]

    unique_actions = np.unique(actions)
    action_to_idx = {a: i for i, a in enumerate(unique_actions)}
    idx_to_action = {i: a for i, a in enumerate(unique_actions)}

    env = BanditEnv(X, actions, rewards, mode="full")

    policy = EpsilonGreedy(
        n_actions=len(unique_actions),
        epsilon=0.1,
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

        assert 0 <= action_idx < len(unique_actions)
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
    print("EpsilonGreedy smoke test passed.")