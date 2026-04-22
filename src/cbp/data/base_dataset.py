import pandas as pd
import numpy as np
from pathlib import Path


class MovieLensDataset:
    def __init__(
        self,
        data_dir: str,
        reward_threshold: float = 4.0,
        min_user_ratings: int = 20,
        max_arms: int = 50,
    ):
        self.data_dir = Path(data_dir)
        self.reward_threshold = reward_threshold
        self.min_user_ratings = min_user_ratings
        self.max_arms = max_arms

    # -------------------------
    # Load raw data
    # -------------------------
    def load_ratings(self):
        df = pd.read_csv(
            self.data_dir / "u.data",
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        return df

    def load_users(self):
        return pd.read_csv(
            self.data_dir / "u.user",
            sep="|",
            names=["user_id", "age", "gender", "occupation", "zip"],
        )

    def load_items(self):
        cols = [
            "item_id", "title", "release", "video_release", "url",
            "unknown", "action", "adventure", "animation", "children",
            "comedy", "crime", "documentary", "drama", "fantasy",
            "film_noir", "horror", "musical", "mystery", "romance",
            "sci_fi", "thriller", "war", "western"
        ]
        return pd.read_csv(
            self.data_dir / "u.item",
            sep="|",
            names=cols,
            encoding="latin-1"
        )

    # -------------------------
    # Preprocessing
    # -------------------------
    def preprocess(self, df):
        df = df.copy()

        # binary reward
        df["reward"] = (df["rating"] >= self.reward_threshold).astype(int)

        return df

    def filter(self, df):
        # filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= self.min_user_ratings].index
        df = df[df["user_id"].isin(valid_users)]

        # limit items (arms)
        top_items = df["item_id"].value_counts().head(self.max_arms).index
        df = df[df["item_id"].isin(top_items)]

        return df.reset_index(drop=True)

    # -------------------------
    # Feature engineering
    # -------------------------
    def build_features(self, df):
        users = self.load_users()
        items = self.load_items()

        df = df.merge(users, on="user_id", how="left")
        df = df.merge(items, on="item_id", how="left")

        # encode gender
        df["gender"] = df["gender"].map({"M": 0, "F": 1})

        # one-hot occupation
        df = pd.get_dummies(df, columns=["occupation"], prefix="occ")

        return df

    # -------------------------
    # Context construction
    # -------------------------
    def build_context(self, df):
        df = df.copy()

        # user features
        user_features = ["age", "gender"]
        user_features += [col for col in df.columns if col.startswith("occ_")]

        # item features (genres)
        item_features = [
            "unknown", "action", "adventure", "animation", "children",
            "comedy", "crime", "documentary", "drama", "fantasy",
            "film_noir", "horror", "musical", "mystery", "romance",
            "sci_fi", "thriller", "war", "western"
        ]

        context_cols = user_features + item_features

        X = df[context_cols].fillna(0).values.astype(np.float32)
        actions = df["item_id"].values
        rewards = df["reward"].values

        return X, actions, rewards, context_cols

    # -------------------------
    # Full pipeline
    # -------------------------
    def load_bandit_data(self):
        df = self.load_ratings()
        df = self.preprocess(df)
        df = self.filter(df)
        df = self.build_features(df)

        X, actions, rewards, context_cols = self.build_context(df)

        return {
            "X": X,
            "actions": actions,
            "rewards": rewards,
            "feature_names": context_cols,
            "df": df,
        }
    

# -------------------------
# Smoke test
# -------------------------

if __name__ == "__main__":
    import numpy as np

    print("Running MovieLensDataset smoke test...")

    data_path = "data/raw/ml-100k"  # adjust if needed

    ds = MovieLensDataset(
        data_dir=data_path,
        reward_threshold=4.0,
        min_user_ratings=20,
        max_arms=50,
    )

    out = ds.load_bandit_data()

    X = out["X"]
    actions = out["actions"]
    rewards = out["rewards"]
    df = out["df"]
    feature_names = out["feature_names"]

    # ---- checks
    assert X.ndim == 2
    assert len(actions) == X.shape[0]
    assert len(rewards) == X.shape[0]
    assert len(feature_names) == X.shape[1]
    assert set(np.unique(rewards)).issubset({0, 1})
    assert np.isfinite(X).all()

    # ---- stats
    print(f"Samples: {X.shape[0]}")
    print(f"Context dim: {X.shape[1]}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(f"Unique items: {df['item_id'].nunique()}")
    print(f"Reward mean: {rewards.mean():.4f}")

    print("Sample:")
    print("X[0][:10]:", X[0][:10])
    print("action[0]:", actions[0])
    print("reward[0]:", rewards[0])

    print("Smoke test passed.")