"""
Functions for visualizing poetry analysis results.

This module contains plotting utilities that take processed data
(e.g., Author objects, metrics DataFrames) and produce visual
representations such as boxplots, histograms, and trend charts.

All functions are designed to be called from notebooks or scripts
after data has been scraped and saved, and should not perform
any scraping or heavy computation themselves.

"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from poetry_project.models import Author


def boxplot_poem_lengths(authors: list[Author]) -> None:
    """
    Plot a boxplot of poem lengths per author, sorted by birth year.

    Args:
        authors: List of Author objects.
    """
    # Flatten into DataFrame
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

    # Sort authors by birth year
    author_order = (
        df_poems.groupby("author")["birth_year"]
        .first()
        .sort_values()
        .index
    )

    # Add labels with birth year
    label_map = {
        author: f"{author} ({df_poems.loc[df_poems['author'] == author, 'birth_year'].iloc[0]})"
        for author in author_order
    }
    df_poems["author_label"] = df_poems["author"].map(label_map)

    # Plot
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
