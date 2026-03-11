import argparse
from pathlib import Path
from evaluationFunctions import evaluation_metrics, add_missing_files

def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Metrics on masks directly against a single Ground Truth folder")
    
    # Paths
    parser.add_argument('--gt-path', type=str, required=True, help='Select ground truth folder containing the GT images')
    parser.add_argument('--predictions-path', type=str, required=True, help='Select model predictions folder')
    parser.add_argument('--yaml-path', type=str, default=None, help='Select yaml training file for multi shape transformations')
    parser.add_argument('--out-name', type=str, default='Evaluation_output', help='Output folder name (optional)')
    
    # Model and Class configurations
    parser.add_argument('--model-dims', type=int, choices=[2, 3], default=2, help='Spatial model dimension (2D or 3D)')
    parser.add_argument('--eval-class', type=int, default=1, help='Evaluation Class')
    
    # Booleans / Flags
    parser.add_argument('--no-verify-images', action='store_true', help='Pass this flag to SKIP verifying missing images')

    args = parser.parse_args()

    # Configuration mapping
    yaml_path = args.yaml_path
    model_dims = args.model_dims
    eval_class = args.eval_class
    verify_images = not args.no_verify_images

    # Paths setup
    selected_gt = Path(args.gt_path)
    base_prediction_path = Path(args.predictions_path)
    
    if not selected_gt.is_dir():
        raise ValueError(f"Ground truth path {selected_gt} is not a valid directory.")
    
    direct_images = sorted(base_prediction_path.glob("*.tiff")) + sorted(base_prediction_path.glob('*.tif'))
    folders_to_process = []
    
    if direct_images:
        print(f"Single model detected at {base_prediction_path.name}")
        folders_to_process.append(base_prediction_path)
    else:
        subdirs = [d for d in base_prediction_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if subdirs:
            print(f"Batch mode detected. Found {len(subdirs)} subfolders in {base_prediction_path.name}:")
            for d in subdirs:
                 print(f" - {d.name}")
            folders_to_process = sorted(subdirs)
        else:
            raise ValueError(f"No .tif/.tiff files or subfolders found in {base_prediction_path}.")

    print(f"###################################### Using single Ground Truth folder: {selected_gt.name} ######################################")
    
    for current_pred_folder in folders_to_process:
        print(f"\n=================================================================================")
        print(f"PROCESSING PREDICTION MODEL: {current_pred_folder.name}")
        print(f"=================================================================================")
        
        files_prediction = sorted(current_pred_folder.glob("*.tiff"))
        files_prediction.extend(list(current_pred_folder.glob('*.tif')))
        
        if not files_prediction:
            print(f"WARNING: No images found in {current_pred_folder.name}. Skipping.")
            continue
        
        output_path = current_pred_folder.parent / args.out_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Found {len(files_prediction)} prediction files.")
        if verify_images: 
            print('--- Looking for missing annotations ---')
            add_missing_files(files_prediction, selected_gt)

        print('--- Generating evaluation metrics ---')
        
        # Calling the newly updated evaluation_metrics function
        evaluation_metrics(
            selected_predictions=current_pred_folder,
            files_prediction=files_prediction,
            output_path=output_path,
            selected_gt=selected_gt,
            eval_class=eval_class,
            yaml_path=yaml_path,
            model_dims=model_dims
        )
        print(f"Completed: {current_pred_folder.name}")

    print('\n###################################### All evaluations completed ############################################')
    print(f'Evaluation results saved at {base_prediction_path.parent if direct_images else base_prediction_path} / {args.out_name}')

if __name__ == '__main__':
    main()