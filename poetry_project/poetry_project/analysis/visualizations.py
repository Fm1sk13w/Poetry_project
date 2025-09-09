"""
Visualization utilities for poetry analysis.

Provides plotting functions for exploring author- and poem-level
metrics, including distributions, trends, correlations, and
linguistic features such as adjective usage.

Author: PoetryProjectBot
"""

from __future__ import annotations
from typing import List, Optional
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from poetry_project.models import Author
from poetry_project.utils.linguistic_utils import adjectives_plus_adverbs_ratio
from poetry_project.utils.analysis_utils import detect_trendy_words

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

import pandas as pd
import matplotlib.pyplot as plt
from typing import List

def plot_poems_by_birth_year(authors: List["Author"], bins: int = 10):
    """
    Plot the number of poems in the dataset grouped by birth_year bins.

    Args:
        authors: List of Author objects with .birth_year and .poems.
        bins: Number of bins to split birth_year into.
    """
    # Build DataFrame of (birth_year, poem_count)
    data = [
        (int(a.birth_year), len(a.poems))
        for a in authors
        if a.birth_year and str(a.birth_year).isdigit() and a.poems
    ]
    if not data:
        raise ValueError("No valid birth_year and poem data found.")

    df = pd.DataFrame(data, columns=["birth_year", "poem_count"])

    # Bin by birth_year
    df["year_bin"] = pd.cut(df["birth_year"], bins=bins)

    # Aggregate total poems per bin
    poems_per_bin = df.groupby("year_bin", observed=False)["poem_count"].sum()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    poems_per_bin.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")

    ax.set_title(f"Number of Poems by Poets' Birth Year (binned into {bins} groups)", fontsize=14)
    ax.set_xlabel("Birth Year Bin")
    ax.set_ylabel("Number of Poems")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
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

def compare_poets_radar(
    metrics_df: pd.DataFrame,
    poet_a: str,
    poet_b: str = None,
    metrics: list = None,
    title: str = None,
    fill_alpha: float = 0.25,
    show_average: bool = False
):
    """
    Create a normalized radar chart comparing two poets (or poet vs average),
    with optional third surface for dataset average.

    All metrics are min-max scaled to [0, 1] across the dataset so that
    differences in scale (e.g., poem length vs ratios) don't distort the shape.

    Args:
        metrics_df: DataFrame with 'author' column and metric columns.
        poet_a: Name of the first poet (always shown).
        poet_b: Name of the second poet. If None, compares poet_a to dataset average.
        metrics: List of metric column names to compare. If None, uses all numeric columns
                 except 'birth_year' and 'poem_count'.
        title: Optional chart title.
        fill_alpha: Transparency for filled areas.
        show_average: If True, always plot dataset average as a third surface.
    """
    # Default metrics if not provided
    if metrics is None:
        metrics = [
            col for col in metrics_df.select_dtypes(include=[np.number]).columns
            if col not in ("birth_year", "poem_count")
        ]

    # Min-max normalize metrics to [0, 1]
    norm_df = metrics_df.copy()
    for col in metrics:
        col_min, col_max = norm_df[col].min(), norm_df[col].max()
        if col_max > col_min:
            norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
        else:
            norm_df[col] = 0.0  # constant column

    # Get poet A's values
    poet_a_row = norm_df.loc[norm_df["author"] == poet_a, metrics]
    if poet_a_row.empty:
        raise ValueError(f"Poet '{poet_a}' not found in metrics_df.")
    poet_a_values = poet_a_row.iloc[0].values

    # Get poet B's values or average
    if poet_b:
        poet_b_row = norm_df.loc[norm_df["author"] == poet_b, metrics]
        if poet_b_row.empty:
            raise ValueError(f"Poet '{poet_b}' not found in metrics_df.")
        poet_b_values = poet_b_row.iloc[0].values
        label_b = poet_b
    else:
        poet_b_values = norm_df[metrics].mean().values
        label_b = "Dataset Average"

    # Dataset average (for optional third surface)
    avg_values = norm_df[metrics].mean().values

    # Number of variables
    num_vars = len(metrics)

    # Angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    # Values for plotting (repeat first value to close the shape)
    values_a = poet_a_values.tolist() + poet_a_values[:1].tolist()
    values_b = poet_b_values.tolist() + poet_b_values[:1].tolist()
    values_avg = avg_values.tolist() + avg_values[:1].tolist()

    # Create polar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], metrics, fontsize=10)

    # Draw ylabels
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
    plt.ylim(0, 1)

    # Plot Poet A
    ax.plot(angles, values_a, color="tab:blue", linewidth=2, label=poet_a)
    ax.fill(angles, values_a, color="tab:blue", alpha=fill_alpha)

    # Plot Poet B / Average
    ax.plot(angles, values_b, color="tab:orange", linewidth=2, label=label_b)
    ax.fill(angles, values_b, color="tab:orange", alpha=fill_alpha)

    # Optional third surface for dataset average
    if show_average and (poet_b is not None):
        ax.plot(angles, values_avg, color="tab:green", linewidth=2, label="Dataset Average")
        ax.fill(angles, values_avg, color="tab:green", alpha=fill_alpha)

    # Title & legend
    if title:
        plt.title(title, size=14, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.show()


def plot_word_trend(word: str, freq_df: pd.DataFrame, bin_edges):
    """
    Plot relative frequency of a word across time bins.

    Args:
        word: The word to plot.
        freq_df: DataFrame from detect_trendy_words (with relative frequencies per bin).
        bin_edges: List of bin edges from detect_trendy_words.
    """
    if word not in freq_df.index:
        raise ValueError(f"Word '{word}' not found in frequency data.")

    bins = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
    values = freq_df.loc[word].iloc[:len(bins)].values  # exclude std_dev column

    plt.figure(figsize=(8, 5))
    plt.plot(bins, values, marker="o", linewidth=2)
    plt.title(f"Usage of '{word}' over time")
    plt.xlabel("Time bin (by birth_year)")
    plt.ylabel("Relative frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def plot_trend_with_lowess(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue: str = None,
    frac: float = 0.3,
    save_path: str = None
) -> None:
    """
    Plot a scatterplot with LOWESS smoothing to visualize non-linear trends.

    Args:
        df: DataFrame containing the data.
        x_col: Column name for x-axis.
        y_col: Column name for y-axis.
        hue: Optional column name for grouping by color.
        frac: Fraction of data used for each LOWESS fit (controls smoothness).
        save_path: Optional path to save the plot.
    """
    plt.figure(figsize=(10, 6))

    if hue:
        unique_groups = df[hue].dropna().unique()
        palette = sns.color_palette("tab10", len(unique_groups))
        for color, group in zip(palette, unique_groups):
            subset = df[df[hue] == group]
            sns.scatterplot(data=subset, x=x_col, y=y_col, color=color, alpha=0.6, label=group)
            smoothed = lowess(subset[y_col], subset[x_col], frac=frac, return_sorted=True)
            plt.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, color="steelblue", alpha=0.6)
        smoothed = lowess(df[y_col], df[x_col], frac=frac, return_sorted=True)
        plt.plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=2)

    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.title(f"LOWESS Trend of {y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_trend_with_rolling_average(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue: str = None,
    window: int = 10,
    save_path: str = None
) -> None:
    """
    Plot a scatterplot with a rolling average trend line.

    Args:
        df: DataFrame containing the data.
        x_col: Column name for x-axis (should be sortable, e.g., year).
        y_col: Column name for y-axis (numeric metric).
        hue: Optional column name for grouping by color.
        window: Window size for rolling average (in number of points).
        save_path: Optional path to save the plot.
    """
    # Sort by x_col to ensure rolling works correctly
    df_sorted = df.sort_values(by=x_col)

    plt.figure(figsize=(10, 6))

    if hue:
        # Plot each group separately
        unique_groups = df_sorted[hue].dropna().unique()
        palette = sns.color_palette("tab10", len(unique_groups))
        for color, group in zip(palette, unique_groups):
            subset = df_sorted[df_sorted[hue] == group]
            sns.scatterplot(data=subset, x=x_col, y=y_col, color=color, alpha=0.6, label=group)
            rolling_avg = subset[y_col].rolling(window=window, center=True).mean()
            plt.plot(subset[x_col], rolling_avg, color=color, linewidth=2)
    else:
        # Scatter all points
        sns.scatterplot(data=df_sorted, x=x_col, y=y_col, color="steelblue", alpha=0.6)
        rolling_avg = df_sorted[y_col].rolling(window=window, center=True).mean()
        plt.plot(df_sorted[x_col], rolling_avg, color="red", linewidth=2, label=f"{window}-point rolling avg")

    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.title(f"Rolling Average Trend of {y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()