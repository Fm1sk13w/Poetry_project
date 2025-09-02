from setuptools import setup, find_packages

setup(
    name="poetry_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "beautifulsoup4",
        "pronouncing",
        # Add others as needed
    ],
    python_requires=">=3.9",
)
