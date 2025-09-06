"""
Visualization utilities for poetry analysis.

Provides plotting functions for exploring author- and poem-level
metrics, including distributions, trends, correlations, and
linguistic features such as adjective usage.

Author: PoetryProjectBot
"""

from __future__ import annotations
from typing import List, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from poetry_project.models import Author
from poetry_project.utils.linguistic_utils import adjectives_plus_adverbs_ratio

# ---------------------------------------------------------------------------
# Global style settings
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper")


# ---------------------------------------------------------------------------
# Core plotting functions
# ---------------------------------------------------------------------------

def boxplot_poem_lengths(authors: List[Author], save_path: Optional[str] = None) -> None:
    """
    Plot a boxplot of poem lengths per author, sorted by birth year.
    """
    poem_records = [
        {
            "author": author.name,
            "birth_year": int(author.birth_year),
            "poem_length": poem.number_of_words
        }
        for author in authors
        if author.birth_year and author.birth_year.isdigit()
        for poem in author.poems
    ]
    df_poems = pd.DataFrame(poem_records)

    author_order = (
        df_poems.groupby("author")["birth_year"]
        .first()
        .sort_values()
        .index
    )

    label_map = {
        author: f"{author} ({df_poems.loc[df_poems['author'] == author, 'birth_year'].iloc[0]})"
        for author in author_order
    }
    df_poems["author_label"] = df_poems["author"].map(label_map)

    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=df_poems,
        x="author_label",
        y="poem_length",
        order=[label_map[a] for a in author_order],
        showfliers=False
    )
    plt.xticks(rotation=90)
    plt.xlabel("Author (Birth Year)")
    plt.ylabel("Poem length (words)")
    plt.title("Distribution of Poem Lengths by Author, Sorted by Birth Year")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_trend_with_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot a scatterplot with regression line to visualize trends.
    """
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=df,
        x=x_col,
        y=y_col,
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "red"},
        ci=95
    )
    if hue:
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue=hue,
            palette="tab10",
            alpha=0.7,
            legend="brief"
        )
        plt.legend(title=hue, loc="best")
    plt.title(f"Trend of {y_col} vs {x_col}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    save_path: Optional[str] = None
) -> None:
    """
    Plot a heatmap of correlations between numeric columns.
    """
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr(method=method)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        cbar=True
    )
    plt.title(f"{method.title()} Correlation Matrix")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_metric_distribution(
    df: pd.DataFrame,
    metric_col: str,
    bins: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Plot the distribution of a single numeric metric.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[metric_col].dropna(), bins=bins, kde=True)
    plt.xlabel(metric_col.replace("_", " ").title())
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {metric_col.replace('_', ' ').title()}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def scatter_authors_by_metrics(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    top_n: int = 5,
    save_path: Optional[str] = None
) -> None:
    """
    Scatter plot of authors by two metrics, highlighting top_n farthest from origin
    after min-max scaling. Top 3 authors are colored gold, silver, and bronze;
    others in top_n are shaded green.

    Args:
        df: DataFrame with 'author', x_col, and y_col columns.
        x_col: Metric for x-axis.
        y_col: Metric for y-axis.
        top_n: Number of top authors to highlight.
        save_path: Optional path to save the plot.
    """
    # Drop missing values
    plot_df = df.dropna(subset=[x_col, y_col]).copy()

    # Min-max scale both metrics to [0, 1]
    x_min, x_max = plot_df[x_col].min(), plot_df[x_col].max()
    y_min, y_max = plot_df[y_col].min(), plot_df[y_col].max()

    plot_df["x_scaled"] = (plot_df[x_col] - x_min) / (x_max - x_min)
    plot_df["y_scaled"] = (plot_df[y_col] - y_min) / (y_max - y_min)

    # Compute Euclidean distance from origin
    plot_df["distance"] = (plot_df["x_scaled"]**2 + plot_df["y_scaled"]**2) ** 0.5

    # Get top N authors
    top_authors = plot_df.nlargest(top_n, "distance").reset_index(drop=True)

    # Start plot
    plt.figure(figsize=(10, 7))

    # Plot all authors in light gray
    sns.scatterplot(
        data=plot_df,
        x="x_scaled",
        y="y_scaled",
        color="lightgray",
        alpha=0.6,
        s=50,
        edgecolor="black",
        marker="o",
        label="Other authors"
    )

    # Medal colors for top 3
    medal_colors = {
        1: "#FFD700",  # Gold
        2: "#C0C0C0",  # Silver
        3: "#CD7F32"   # Bronze
    }

    # Green shades for ranks > 3
    greens = sns.color_palette("Greens", n_colors=max(top_n - 3, 1) + 2)[2:]

    # Plot top authors
    for rank, (_, row) in enumerate(top_authors.iterrows(), start=1):
        if rank in medal_colors:
            color = medal_colors[rank]
        else:
            green_index = (top_n - rank) if (top_n - 3) > 0 else 0
            color = greens[green_index]

        sns.scatterplot(
            x=[row["x_scaled"]],
            y=[row["y_scaled"]],
            color=color,
            s=100,
            edgecolor="black",
            marker="o",
            label=f"Rank {rank} – {row['author']}"
        )

    # Axis labels and title
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.title(f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}")

    # Grid and legend
    plt.grid(alpha=0.3)
    plt.legend(title="Top authors by scaled distance", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Optional save
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

