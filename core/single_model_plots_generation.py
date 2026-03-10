import argparse
from pathlib import Path
from tqdm import tqdm
from evaluationFunctions import graph_generator

def main():
    parser = argparse.ArgumentParser(description="Run plot generation from CSVs")
    parser.add_argument('--csv-path', type=str, required=True, help="Folder containing the CSV files")
    
    args = parser.parse_args()
    selected_path = Path(args.csv_path)
    
    if not selected_path.exists() or not selected_path.is_dir():
        raise ValueError(f"The path provided does not exist or is not a directory: {selected_path}")
        
    files = [f.name for f in selected_path.glob("*model_eval*.csv")]
    if not files:
        print(f"No files matching '*model_eval*.csv' found in {selected_path}")
        return

    print('###################################### Plots generation starting ############################################')
    for file in tqdm(files):
        graph_generator(selected_path / file)
    print('###################################### Plot generation complete ############################################')

if __name__ == '__main__':
    main()