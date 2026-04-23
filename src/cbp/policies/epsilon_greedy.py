import numpy as np

 
class EpsilonGreedy:
    def __init__(self, n_actions: int, epsilon: float = 0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon

        self.counts = np.zeros(n_actions, dtype=np.int32)
        self.values = np.zeros(n_actions, dtype=np.float32)

    def select_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.values)

    def update(self, action: int, reward: float):
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

    # load data
    ds = MovieLensDataset(data_dir="data/raw/ml-100k")
    data = ds.load_bandit_data()

    X = data["X"]
    actions = data["actions"]
    rewards = data["rewards"]

    # map actions → indices
    unique_actions = np.unique(actions)
    action_to_idx = {a: i for i, a in enumerate(unique_actions)}
    idx_to_action = {i: a for i, a in enumerate(unique_actions)}

    # init env + policy
    env = BanditEnv(X, actions, rewards, mode="full")
    policy = EpsilonGreedy(n_actions=len(unique_actions), epsilon=0.1)

    env.reset()

    total_reward = 0
    steps = 0

    for _ in range(50):  # small test
        context = env.get_context()

        action_idx = policy.select_action()
        action = idx_to_action[action_idx]

        reward, done, info = env.step(action)

        # checks
        assert action_idx >= 0 and action_idx < len(unique_actions)
        assert reward in [0, 1]

        policy.update(action_idx, reward)

        total_reward += reward
        steps += 1

        print(f"Step {steps} | action_idx={action_idx} | reward={reward}")

        if done:
            break

    print(f"Total reward: {total_reward}")
    print("EpsilonGreedy smoke test passed.")