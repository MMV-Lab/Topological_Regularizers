import argparse
from pathlib import Path
from evaluationFunctions import evaluation_metrics, add_missing_files

def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Metrics on masks")
    
    # Paths
    parser.add_argument('--gt-path', type=str, required=True, help='Select ground truth folder')
    parser.add_argument('--predictions-path', type=str, required=True, help='Select model predictions folder')
    parser.add_argument('--yaml-path', type=str, default=None, help='Select yaml training file for multi shape transformations')
    parser.add_argument('--out-name', type=str, default='Evaluation_output', help='Output folder name (optional)')
    
    # Model and Class configurations
    parser.add_argument('--model-dims', type=int, choices=[2, 3], default=2, help='Spatial model dimension (2D or 3D)')
    parser.add_argument('--gt-mode', type=str, choices=['single', 'multiclass'], default='single', help='GT class annotations')
    parser.add_argument('--eval-mode', type=str, choices=['single', 'all'], default='single', help='Evaluation class mode')
    parser.add_argument('--eval-class', type=int, default=1, help='Evaluation Class')
    parser.add_argument('--staple-threshold', type=float, default=0.5, help='STAPLE Threshold')
    
    # Booleans / Flags
    parser.add_argument('--use-dilatation', action='store_true', help='Flag to use dilatation')
    parser.add_argument('--dilation-pixels', type=int, default=3, help='Radius for dilatation')
    parser.add_argument('--dilation-mode', type=str, choices=['disk', 'square'], default='disk', help='Mode for dilatation')
    parser.add_argument('--no-verify-images', action='store_true', help='Pass this flag to SKIP verifying missing images')
    parser.add_argument('--save-mask', action='store_true', help='Flag to save generated masks')

    args = parser.parse_args()

    # Determine yaml/model-dims logic
    yaml_path = args.yaml_path
    model_dims = args.model_dims
    
    selected_gt = Path(args.gt_path)
    base_prediction_path = Path(args.predictions_path)
    
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

    # Evaluation mode
    eval_class = args.eval_class if args.eval_mode == 'single' else 'all'
    verify_images = not args.no_verify_images

    # Annotators files
    anotators_files = [item.name for item in selected_gt.iterdir() if item.is_dir()]
    if len(anotators_files) == 0:
        raise ValueError(f"No ground truth folders found in {selected_gt}.")

    # Dilatation params
    dilatation = True if args.use_dilatation else None
    n_pixels = args.dilation_pixels if args.use_dilatation else None
    kernel_shape = args.dilation_mode if args.use_dilatation else None

    print(f"###################################### { len(anotators_files) } annotators folders found in GT.######################################")
    
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
            add_missing_files(files_prediction, anotators_files, selected_gt)

        print('--- Generating evaluation metrics ---')
        evaluation_metrics(
            anotators_files,
            current_pred_folder,
            files_prediction,
            output_path,
            selected_gt,
            args.save_mask,
            dilatation,
            n_pixels,
            kernel_shape, 
            eval_class, 
            args.staple_threshold,
            args.gt_mode,
            yaml_path,
            model_dims
        )
        print(f"Completed: {current_pred_folder.name}")

    print('\n###################################### All evaluations completed ############################################')
    print(f'Evaluation results saved at {base_prediction_path.parent if direct_images else base_prediction_path} / {args.out_name}')

if __name__ == '__main__':
    main()