from setuptools import setup, find_packages

setup(
    name="poetry_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",          # HTTP requests for scraping and APIs
        "beautifulsoup4",    # HTML parsing
        "pronouncing",       # CMU Pronouncing Dictionary
        "pandas",            # Data handling and analysis
        "matplotlib",        # Plotting
        "seaborn",           # Statistical data visualization
        "scipy",             # Statistical functions (pearsonr, etc.)
        "tqdm",              # Progress bars
        "pyarrow",           # Parquet file support for metrics
    ],
    python_requires=">=3.9",
)
