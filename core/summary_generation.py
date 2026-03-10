import argparse
from pathlib import Path
from evaluationFunctions import generate_statistical_summaries

def main():
    parser = argparse.ArgumentParser(description="Generate statistical summaries from CSVs")
    parser.add_argument('--csv-path', type=str, required=True, help="Folder containing the CSV files")
    
    args = parser.parse_args()
    selected_path = Path(args.csv_path)
    
    if not selected_path.exists() or not selected_path.is_dir():
        raise ValueError(f"The path provided does not exist or is not a directory: {selected_path}")
    
    print('###################################### Summary generation ############################################')
    generate_statistical_summaries(selected_path)
    print('###################################### Summary complete ############################################')

if __name__ == '__main__':
    main()