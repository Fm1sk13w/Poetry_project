# Poetry_project
A Python-powered pipeline for scraping, cleaning, and integrating poetry and author metadata from available web sources. Designed with a data science focus, this project transforms raw textual data into structured datasets for exploratory analysis.

Key features:
- **Automated scraping** of poems from https://poemanalysis.com/ and merging with authors data from https://www.wikidata.org/.
- **Data cleaning & structuring** into a unified format.
- **Linguistic analysis**: rhyme density, adjective/adverb ratios, lexical richness, syllable counts, and more.
- **Trend detection**: time-based patterns, and “trendy” words.
- **Visualization**: plots for trends, radar charts, comparisons, and statistical summaries.

Instruction:
1) Install dependencies
2) Run the scraper script: poetry_project/scripts/fetch_data.py
3) Load and analyze data using tools from poetry_project (see data_analysis/analysis_notebook.ipynb for example)
