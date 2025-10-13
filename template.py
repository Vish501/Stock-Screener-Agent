import os
import logging

from pathlib import Path

def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

    # List of files and directories that need to be created or amended with new files
    list_of_files = [
        "src/StockScreener/__init__.py",
        "src/StockScreener/components/__init__.py",
        "src/StockScreener/utils/__init__.py",
        "src/StockScreener/utils/common.py",
        "src/StockScreener/utils/logger.py",
        "src/StockScreener/config/__init__.py",
        "src/StockScreener/config/configurations.py",
        "src/StockScreener/pipeline/__init__.py",
        "src/StockScreener/entity/__init__.py",
        "src/StockScreener/constants/__init__.py",
        "pyproject.toml",
        "requirements.txt",
        "research/.gitkeep"
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
    