"""
Fetch poetry data and save it locally.
"""

from poetry_project.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline(min_poems=25, max_poems=200)
