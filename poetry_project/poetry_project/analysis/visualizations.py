"""
Visualization utilities for poetry analysis.

This module provides plotting functions for exploring author- and poem-level
metrics, including distributions, trends, and correlations.

Author: PoetryProjectBot
"""

from __future__ import annotations

from typing import List, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from poetry_project.models import Author


def boxplot_poem_lengths(authors: List[Author]) -> None:
    """
    Plot a boxplot of poem lengths per author, sorted by birth year.

    Args:
        authors: List of Author objects.
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
    plt.tight_layout()
    plt.show()


def plot_trend_with_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue: Optional[str] = None
) -> None:
    """
    Plot a scatterplot with regression line to visualize trends.

    Args:
        df: DataFrame containing the data.
        x_col: Column name for the x-axis (independent variable).
        y_col: Column name for the y-axis (dependent variable).
        hue: Optional column name for color grouping.
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
    plt.title(f"Trend of {y_col} vs {x_col}")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson"
) -> None:
    """
    Plot a heatmap of correlations between numeric columns.

    Args:
        df: DataFrame containing numeric columns.
        method: Correlation method ('pearson' or 'spearman').
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
    plt.tight_layout()
    plt.show()


def plot_metric_distribution(
    df: pd.DataFrame,
    metric_col: str,
    bins: int = 20
) -> None:
    """
    Plot the distribution of a single numeric metric.

    Args:
        df: DataFrame containing the metric.
        metric_col: Column name of the metric to plot.
        bins: Number of histogram bins.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[metric_col].dropna(), bins=bins, kde=True)
    plt.xlabel(metric_col.replace("_", " ").title())
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {metric_col.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.show()
