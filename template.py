import os
import logging

from pathlib import Path

def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

    src_file_name = "StockScreener"

    # List of files and directories that need to be created or amended with new files
    list_of_files = [
        f"src/{src_file_name}/__init__.py",
        f"src/{src_file_name}/utils/__init__.py",
        f"src/{src_file_name}/utils/common.py",
        f"src/{src_file_name}/utils/logger.py",
        f"src/{src_file_name}/core/__init__.py",
        f"src/{src_file_name}/core/graph/__init__.py",
        f"src/{src_file_name}/core/graph/stock_screener_graph.py",
        f"src/{src_file_name}/core/graph/nodes/__init__.py",
        f"src/{src_file_name}/core/agents/__init__.py",
        f"src/{src_file_name}/core/llm/__init__.py",
        f"src/{src_file_name}/core/scoring/__init__.py",
        f"src/{src_file_name}/core/retrieval/__init__.py",
        f"src/{src_file_name}/core/data/__init__.py",
        f"src/{src_file_name}/core/prompts/.gitkeep",
        f"src/{src_file_name}/utils/__init__.py",
        f"src/{src_file_name}/pipelines/__init__.py",
        "artifacts/prices/.gitkeep",
        "artifacts/fundamentals/.gitkeep",
        "artifacts/sentiment/.gitkeep",
        "artifacts/embeddings/.gitkeep",
        "artifacts/cache_index.json",
        "artifacts/benchmarks/.gitkeep",
        "params/thresholds.yaml",
        "config/config.yaml",
        "pyproject.toml",
        "requirements.txt",
        "research/.gitkeep",
        "app/main.py",
        "app/componets/chat_window.py",
        "app/utils/memory.py",
        "app/utils/prompts.py",
        "app/utils/formatters.py",
        f"src/{src_file_name}/core/data/cache_manager.py",
        f"src/{src_file_name}/core/data/data_loader.py",
        f"src/{src_file_name}/core/data/feature_engineer.py",
        f"src/{src_file_name}/core/data/schemas.py",
        f"src/{src_file_name}/pipelines/fetch_prices.py",           # Fetch from Yahoo/AlphaVantage
        f"src/{src_file_name}/pipelines/fetch_fundamentals.py",     # Financial ratios and balance sheets
        f"src/{src_file_name}/pipelines/fetch_sentiment.py",        # NewsAPI, Reddit, or Twitter fetch
        f"src/{src_file_name}/pipelines/fetch_benchmarks.py",       # Runs all fetchers and updates cache_index.json
        f"src/{src_file_name}/pipelines/update_cache.py",           # Optional cron-like update orchestration
    ]

    # Iterate through each file path
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, _ = os.path.split(filepath)
        
        # Create the directory if it doesn't already exist
        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f'Creating directory: {filedir}')
        
        # Create the file if it doesn't exist or is currently empty
        if not filepath.exists() or filepath.stat().st_size == 0:
            filepath.touch()
            logging.info(f"Created empty file: {filepath}")
        else:
            logging.info(f"File already exists: {filepath}")


if __name__ == "__main__":
    main()
    