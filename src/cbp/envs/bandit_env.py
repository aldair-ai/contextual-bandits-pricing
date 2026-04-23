import numpy as np


class BanditEnv:
    def __init__(self, X, actions, rewards, mode="full"):
        """
        mode:
            - "full"   → reward always available (recommended now)
            - "replay" → reward only if action == logged_action
        """
        self.X = X
        self.logged_actions = actions
        self.rewards = rewards

        self.n = len(X)
        self.mode = mode

        self.t = 0

    def reset(self):
        self.t = 0

    def get_context(self):
        return self.X[self.t]

    def get_logged_action(self):
        return self.logged_actions[self.t]

    def get_logged_reward(self):
        return self.rewards[self.t]

    def step(self, action):
        """
        Returns:
            reward, done, info
        """
        if self.mode == "full":
            reward = self.rewards[self.t]

        elif self.mode == "replay":
            if action == self.logged_actions[self.t]:
                reward = self.rewards[self.t]
            else:
                reward = None  # no feedback

        else:
            raise ValueError("Invalid mode")

        info = {
            "t": self.t,
            "logged_action": self.logged_actions[self.t],
        }

        self.t += 1
        done = self.t >= self.n

        return reward, done, info
    

# -------------------------
# Smoke test
# -------------------------

if __name__ == "__main__":
    print("Running BanditEnv smoke test...")

    from src.cbp.data.base_dataset import MovieLensDataset

    # load data
    ds = MovieLensDataset(data_dir="data/raw/ml-100k")
    data = ds.load_bandit_data()

    X = data["X"]
    actions = data["actions"]
    rewards = data["rewards"]

    # init env
    env = BanditEnv(X, actions, rewards, mode="full")

    env.reset()

    # basic checks
    assert len(X) == len(actions) == len(rewards)

    total_reward = 0
    steps = 0

    for _ in range(10):
        context = env.get_context()

        # random policy
        action = np.random.choice(actions)

        reward, done, info = env.step(action)

        assert context.shape[0] == X.shape[1]
        assert reward in [0, 1]

        total_reward += reward
        steps += 1

        print(f"Step {steps} | reward={reward} | logged_action={info['logged_action']}")

        if done:
            break

    print(f"Total reward (10 steps): {total_reward}")
    print("BanditEnv smoke test passed.")
