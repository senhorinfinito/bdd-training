import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame


class VisDataset:
    def __init__(self, config, train_data=[], val_data=[], isfiltered=False):
        self.config = config
        self.train_df = DataFrame(train_data)
        self.val_df = DataFrame(val_data)
        self.vis_dir = config.get("paths", {}).get("vis_dir", "./cfg/data/insights")
        if isfiltered:
            self.vis_dir = os.path.join(self.vis_dir, "filtered")
        os.makedirs(self.vis_dir, exist_ok=True)

    def _counts(self, df, col):
        return df[col].value_counts().to_dict() if col in df.columns else {}

    def classwise(self, istrain=True):
        df = self.train_df if istrain else self.val_df
        return self._counts(df, "category")

    def weatherwise(self, istrain=True):
        df = self.train_df if istrain else self.val_df
        return self._counts(df, "weather")

    def scenewise(self, istrain=True):
        df = self.train_df if istrain else self.val_df
        return self._counts(df, "scene")

    def plot_bar(self, details, title="", save_name="plot.png"):
        plt.figure(figsize=(10, 5))
        colors = plt.cm.tab20.colors
        plt.bar(
            details.keys(),
            details.values(),
            color=[colors[i % len(colors)] for i in range(len(details))],
        )
        plt.xlabel("Categories")
        plt.ylabel("Counts")
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, save_name))
        plt.close()

    def plot_all(self, istrain=False):
        df_type = "train" if istrain else "val"
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        plot_data = [
            (self.classwise(istrain), "Class-wise Distribution"),
            (self.weatherwise(istrain), "Weather-wise Distribution"),
            (self.scenewise(istrain), "Scene-wise Distribution"),
        ]

        for ax, (details, title) in zip(axes, plot_data):
            colors = plt.cm.tab20.colors
            ax.bar(
                details.keys(),
                details.values(),
                color=[colors[i % len(colors)] for i in range(len(details))],
            )
            ax.set_xlabel("Categories")
            ax.set_ylabel("Counts")
            ax.set_title(title)
            ax.set_xticks(range(len(details)))
            ax.set_xticklabels(details.keys(), rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, f"{df_type}_combined_distributions.png"))
        plt.close()

    def compare(
        self,
        train_details,
        val_details,
        title="Comparison",
        save_name="train_val_comparison.png",
    ):
        categories = list(set(train_details.keys()) | set(val_details.keys()))
        categories.sort()

        train_counts = [train_details.get(cat, 0) for cat in categories]
        val_counts = [val_details.get(cat, 0) for cat in categories]

        x = np.arange(len(categories))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width / 2, train_counts, width, label="Train", color="skyblue")
        plt.bar(x + width / 2, val_counts, width, label="Val", color="orange")

        plt.xlabel("Categories")
        plt.ylabel("Counts")
        plt.title(title)
        plt.xticks(x, categories, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, save_name))
        plt.close()

    def compare_all(self):
        # Compare classwise
        self.compare(
            self.classwise(istrain=True),
            self.classwise(istrain=False),
            title="Class-wise Train vs Val Distribution",
            save_name="classwise_train_vs_val.png",
        )
        # Compare weatherwise
        self.compare(
            self.weatherwise(istrain=True),
            self.weatherwise(istrain=False),
            title="Weather-wise Train vs Val Distribution",
            save_name="weatherwise_train_vs_val.png",
        )
        # Compare scenewise
        self.compare(
            self.scenewise(istrain=True),
            self.scenewise(istrain=False),
            title="Scene-wise Train vs Val Distribution",
            save_name="scenewise_train_vs_val.png",
        )
