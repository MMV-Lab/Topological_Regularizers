import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path
import numpy as np
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from tqdm.auto import tqdm
import bioio_tifffile
from skimage.morphology import dilation, disk, square
from scipy.ndimage import label, binary_erosion
import pandas as pd
import matplotlib.pyplot as plt
import logging
import torch
import monai
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
import yaml
from monai.transforms import Spacing, Resize
from monai.data import MetaTensor

logging.getLogger("bioio").setLevel(logging.ERROR)

##################################################### metrics computations functions ################################################################
def to_monai_tensor(matrix):
    """
    Converts a numpy array of shape (Z, Y, X) or (Y, X) into 
    a MONAI compatible PyTorch tensor of shape (B, C, ...).
    """
    return torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def extract_boundary(mask, min_boundary_width=1, dilation_ratio=0.02):
    """
    Extracts the boundary of a binary mask using morphological operations.
    Supports both 2D and 3D robustly.
    """
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
        
    dims = mask.shape
    diag = np.sqrt(sum(d**2 for d in dims))
    d = max(int(round(dilation_ratio * diag)), min_boundary_width)
    
    eroded = binary_erosion(mask, iterations=d)
    boundary = mask.astype(bool) & ~eroded
    return boundary


def boundary_iou(y_pred, y_true):
    """
    Calculates Boundary Intersection over Union (Boundary Crispness).
    """
    bound_pred = extract_boundary(y_pred)
    bound_true = extract_boundary(y_true)
    
    intersection = np.sum(bound_pred & bound_true)
    union = np.sum(bound_pred | bound_true)
    
    if union == 0:
        return 1.0 if np.sum(bound_pred) == 0 and np.sum(bound_true) == 0 else 0.0
    return intersection / union


def overall_contour_agreement(y_pred, y_true):
    """
    Calculates the Overall Contour Agreement (Boundary Dice / F1 Score).
    """
    bound_pred = extract_boundary(y_pred)
    bound_true = extract_boundary(y_true)
    
    intersection = np.sum(bound_pred & bound_true)
    sum_bounds = np.sum(bound_pred) + np.sum(bound_true)
    
    if sum_bounds == 0:
        return 1.0 if np.sum(bound_pred) == 0 and np.sum(bound_true) == 0 else 0.0
    return 2.0 * intersection / sum_bounds


def hausdorff_computing(pr, m_pred, m_gt):
    metric_fn = HausdorffDistanceMetric(percentile=pr, include_background=True, get_not_nans=False)
    m_pred_t = to_monai_tensor(m_pred)
    m_gt_t = to_monai_tensor(m_gt)
    
    if m_pred_t.sum() == 0 and m_gt_t.sum() == 0:
        return 0.0
    elif m_pred_t.sum() == 0 or m_gt_t.sum() == 0:
        return 100.0
    
    metric_fn(y_pred=m_pred_t, y=m_gt_t)
    hd_sym = metric_fn.aggregate().item()
    metric_fn.reset()
    return hd_sym


def msd_computing(m_pred, m_gt):
    """
    Calculates Symmetric Mean Surface Distance using MONAI.
    """
    metric_fn = SurfaceDistanceMetric(include_background=True, symmetric=True, get_not_nans=False)
    m_pred_t = to_monai_tensor(m_pred)
    m_gt_t = to_monai_tensor(m_gt)
    
    if m_pred_t.sum() == 0 and m_gt_t.sum() == 0:
        return 0.0 
    elif m_pred_t.sum() == 0 or m_gt_t.sum() == 0:
        return 100.0
    
    metric_fn(y_pred=m_pred_t, y=m_gt_t)
    msd_sym = metric_fn.aggregate().item()
    metric_fn.reset()
    return msd_sym


def jaccard_computing(m_pred, m_gt):
    m_pred_bool = m_pred.astype(bool)
    m_gt_bool = m_gt.astype(bool)
    
    intersection = np.sum(m_pred_bool & m_gt_bool)
    union = np.sum(m_pred_bool | m_gt_bool)
    return intersection / union if union != 0 else 1.0


def binarizar_matrix(matrix):
    if len(matrix.shape) == 2:
        matrix = matrix[None,...]
    
    if len(matrix.shape) != 3:
        raise ValueError(f"Image dims {matrix.shape} but ZYX are required")
    
    binary = []
    for zz in range(matrix.shape[0]):     
        mask = matrix[zz] != 0
        matrix_bin = np.zeros_like(matrix[zz], dtype=np.uint8)
        matrix_bin[mask] = 1
        binary.append(matrix_bin.astype(np.uint8))

    return np.stack(binary, axis=0).astype(np.uint8)


def thicken_segmentation_skimage(binary_matrix, n_pixels, kernel_shape):
    if len(binary_matrix.shape) == 2:
        binary_matrix = binary_matrix[None,...]
    
    if len(binary_matrix.shape) != 3:
        raise ValueError(f"Image dims {binary_matrix.shape} but ZYX are required")    

    if kernel_shape == 'disk':
        selem = disk(n_pixels)
    elif kernel_shape == 'square':
        side_length = 2 * n_pixels + 1
        selem = square(side_length)
    else:
        raise ValueError("The kernel_shape must be 'square' or 'disk'.")
    
    thickened_segmentation = []
    for zz in range(binary_matrix.shape[0]):
        thickened_segmentation.append(dilation(binary_matrix[zz], footprint=selem).astype(np.uint8))

    return np.stack(thickened_segmentation, axis=0)


def read_file(fn):
    try:
        img_pred = BioImage(fn, reader=bioio_tifffile.Reader).get_image_data("ZYX", C=0, T=0)
    except Exception:
        try:
            img_pred = BioImage(fn).get_image_data("ZYX", C=0, T=0)  
        except Exception:
            raise ValueError("Error at reading time.")
    return img_pred


def extract_class(matrix, target_class):
    if len(matrix.shape) == 2:
        matrix = matrix[None,...]
    
    if len(matrix.shape) != 3:
        raise ValueError(f"Image dims {matrix.shape} but ZYX are required")
    
    if isinstance(target_class, int):
        class_f = (matrix == target_class).astype(np.uint8)
    elif isinstance(target_class, str):
        class_f = (matrix > 0).astype(np.uint8)
    else:
        raise ValueError(f"Check the target class str/int input is required but {target_class}-{type(target_class)} is given.")         
    
    return class_f


def count_components(matrix):
    if len(matrix.shape) == 2:
        matrix = matrix[None,...]
    if len(matrix.shape) != 3:
        raise ValueError(f"Image dims {matrix.shape} but ZYX are required")
    _, num_objects = label(matrix)
    return num_objects


def add_missing_files(files_prediction, selected_gt):
    for fn in tqdm(files_prediction, desc="Looking for missing annotations"):
        stem_file = fn.stem.replace('_segPred','') 
        ext = fn.suffix
        tiff_file = selected_gt / f"{stem_file}.tiff"
        tif_file = selected_gt / f"{stem_file}.tif"
        
        if not (tiff_file.exists() or tif_file.exists()):
            out_img_file = f"{stem_file}{ext}"
            save_path = selected_gt / out_img_file
            img_pred = read_file(fn)
            zero_pred = np.zeros(img_pred.shape)
            OmeTiffWriter.save(data=zero_pred, uri=save_path, dim_order="ZYX")


def extract_params(yaml_path, dim=None):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    preprocess_transforms = config.get("data", {}).get("preprocess", [])

    extracted = {
        "pixdim": None,
        "spacing_mode": None,
        "spatial_size": None,
        "resize_mode": None,
    }
    
    spatial_dim = config.get("model",{}).get("net",{}).get("params",{}).get("spatial_dims",None) if dim is None else dim    

    use_S = False
    use_R = False
    for transform in preprocess_transforms:
        func_name = transform.get("func_name", "")
        params = transform.get("params", {})

        if func_name == "Spacingd":
            use_S = True
            extracted["pixdim"] = params.get("pixdim", None)
            if spatial_dim == 2 and len(extracted["pixdim"]) == 2:
                extracted["pixdim"].insert(0,1) 
                  
            mode = params.get("mode", None)
            extracted["spacing_mode"] = "nearest" if isinstance(mode, list) else mode
            
        if func_name == "Resized":
            use_R = True
            extracted["spatial_size"] = params.get("spatial_size", None)
            if spatial_dim == 2 and len(extracted["spatial_size"]) == 2:
                extracted["spatial_size"].insert(0,1)  

            mode = params.get("mode", None)
            extracted["resize_mode"] = "nearest" if isinstance(mode, list) else mode
        
    if not (use_R or use_S):
        raise ValueError(f"Not Resized or Spacingd definition found in the yaml {yaml_path}")

    return extracted, spatial_dim 


def apply_transforms(image_np, config, original_pixdim=None):
    pixdim = config.get("pixdim", None)
    spacing_mode = config.get("spacing_mode", "nearest")
    spatial_size = config.get("spatial_size", None)
    resize_mode = config.get("resize_mode", "nearest")
    tensor = torch.from_numpy(image_np)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0) 
        spatial_dims = 2
        affine = torch.eye(3)
    else:
        tensor = tensor.unsqueeze(0) 
        spatial_dims = 3
        affine = torch.eye(4)

    tensor = MetaTensor(tensor, affine=affine)

    if pixdim is not None:
        if original_pixdim is None:
            original_pixdim = [1.0] * spatial_dims
        spacing_transform = Spacing(pixdim=pixdim[:spatial_dims], mode=spacing_mode)
        tensor = spacing_transform(tensor)

    if spatial_size is not None:
        resize_transform = Resize(spatial_size=spatial_size[:spatial_dims], mode=resize_mode)
        tensor = resize_transform(tensor)
        
    tensor = tensor.squeeze(0)
    result = tensor.cpu().numpy()
    if np.issubdtype(image_np.dtype, np.integer):
        result = np.rint(result).astype(image_np.dtype)

    return result


def evaluation_metrics(
    selected_predictions,
    files_prediction,
    output_path,
    selected_gt,
    eval_class, 
    yaml_path=None,
    model_dims=None
    ):

    if yaml_path is not None:
        params, _ = extract_params(yaml_path, model_dims) 

    eval_id = str(selected_predictions).split('_')[-1]
    
    image_id = []
    counts_model = []
    counts_gt = []
    all_jaccard = []
    all_hausdorff = []
    all_msd = [] 
    all_biou = []
    all_oca = []

    for fn in tqdm(files_prediction, desc="Evaluation process"):
        stem_file = fn.stem.replace('_segPred','')
        image_id.append(stem_file)

 
        model_pred = binarizar_matrix(extract_class(read_file(fn), eval_class))   

        counts_model.append(count_components(model_pred)) 
        
        tiff_file = selected_gt / f"{stem_file}.tiff"
        tif_file = selected_gt / f"{stem_file}.tif"
        try:
            ann_im = read_file(tiff_file if tiff_file.exists() else tif_file)
        except Exception:
            raise ValueError(f"Error finding {stem_file} in {selected_gt}.")
            
        if yaml_path is None:        
            if model_pred.shape != ann_im.shape:
                raise ValueError(f"Image {stem_file} has different shape for Ground Truth model prediction {model_pred.shape} differ from {ann_im.shape}.")

        ann_im = binarizar_matrix(extract_class(ann_im, eval_class))    
 
        if yaml_path is not None:
            ann_im = apply_transforms(ann_im, params)     

        counts_gt.append(count_components(ann_im)) 
        
        # Computations directly against the single GT
        all_jaccard.append(jaccard_computing(model_pred, ann_im))
        all_hausdorff.append(hausdorff_computing(95, model_pred, ann_im))
        all_msd.append(msd_computing(model_pred, ann_im))
        all_biou.append(boundary_iou(model_pred, ann_im))
        all_oca.append(overall_contour_agreement(model_pred, ann_im))

    data = {
        'image_id': image_id,
        'Ground_Truth': counts_gt,
        'Model': counts_model,
        'Ground_Truth_jaccard': all_jaccard,
        'Ground_Truth_hausdorff': all_hausdorff,
        'Ground_Truth_msd': all_msd,
        'Ground_Truth_biou': all_biou,
        'Ground_Truth_oca': all_oca
    }
    
    df = pd.DataFrame(data)
    
    df['Ground_Truth_dice'] = df['Ground_Truth_jaccard'].apply(lambda j: (2 * j) / (1 + j) if j >= 0 else 0)
    
    # Reorder columns intuitively
    cols_to_keep = ['image_id', 'Ground_Truth', 'Model', 
                    'Ground_Truth_jaccard', 'Ground_Truth_dice', 'Ground_Truth_hausdorff', 
                    'Ground_Truth_msd', 'Ground_Truth_biou', 'Ground_Truth_oca']
    df = df[cols_to_keep]
        
    csv_filename = f'model_evaluation_{eval_id}.csv'
    csv_path = output_path / csv_filename
    df.to_csv(csv_path, index=False)


##################################################### summary plot functions ################################################################
def save_summary_plots(df, folder_path, is_single=True):
    if df.empty:
        return

    label_col = 'Statistic' if 'Statistic' in df.columns else 'id'
    plot_df = df[df[label_col] == 'mean'] if is_single and label_col == 'Statistic' else df

    if len(df.columns) <= 10: # Increased threshold naturally since GT is directly merged
        m_jaccard = [c for c in df.columns if 'jaccard' in c.lower()]
        m_dice = [c for c in df.columns if 'dice' in c.lower()]
        m_hausdorff = [c for c in df.columns if 'hausdorff' in c.lower()]
        m_msd = [c for c in df.columns if 'msd' in c.lower()]
        m_biou = [c for c in df.columns if 'biou' in c.lower()]
        m_oca = [c for c in df.columns if 'oca' in c.lower()]

        metrics_list = []
        if m_jaccard: metrics_list.append(('Jaccard', m_jaccard[0], 'left'))
        if m_dice: metrics_list.append(('Dice', m_dice[0], 'left'))
        if m_biou: metrics_list.append(('BIoU', m_biou[0], 'left'))
        if m_oca: metrics_list.append(('OCA', m_oca[0], 'left'))
        if m_hausdorff: metrics_list.append(('Hausdorff', m_hausdorff[0], 'right'))
        if m_msd: metrics_list.append(('MSD', m_msd[0], 'right'))

        plt.figure(figsize=(12, 7))
        ax1 = plt.gca()
        ax2 = ax1.twinx() 

        n_models = len(plot_df)
        n_metrics = len(metrics_list)
        x = np.arange(n_metrics)
        width = 0.8 / n_models 

        colors = plt.cm.tab20(np.linspace(0, 1, n_models))
        legend_handles = []

        for i, (idx, row) in enumerate(plot_df.iterrows()):
            model_name = row[label_col]
            model_color = colors[i]
            
            for m_idx, (m_label, col_name, side) in enumerate(metrics_list):
                pos = m_idx + (i - n_models/2 + 0.5) * width
                val = row[col_name]
                
                if side == 'left':
                    bar = ax1.bar(pos, val, width, color=model_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                else:
                    bar = ax2.bar(pos, val, width, color=model_color, alpha=0.6, hatch='//', edgecolor='black', linewidth=0.5)
                
                if m_idx == 0:
                    legend_handles.append(plt.Rectangle((0,0),1,1, color=model_color, label=model_name))

        ax1.set_ylabel('Precision Scores (Jaccard / Dice / BIoU / OCA)↑', color='blue', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1.1)
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2.set_ylabel('Distance Metrics (Hausdorff / MSD)↓', color='red', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title('Model Comparison', fontsize=14, pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels([m[0] for m in metrics_list], fontsize=11)
        ax1.set_xlabel('Evaluation Metrics', fontsize=12)
        
        ax1.legend(handles=legend_handles, title="Models", loc='upper left', bbox_to_anchor=(1.15, 1))
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        save_name = folder_path / "summary_plot.png"
        plt.savefig(save_name, dpi=300)
        plt.close()
    else:
        metrics = {
            'Jaccard': '_jaccard',
            'Dice': '_dice',
            'MSD': '_msd',
            'Hausdorff': '_hausdorff',
            'BIoU': '_biou',
            'OCA': '_oca'
        }

        label_col = 'Statistic' if 'Statistic' in df.columns else 'id'
        plot_df = df[df[label_col] == 'mean'] if is_single else df

        num_curves = len(plot_df)
        colors = plt.cm.tab20(np.linspace(0, 1, num_curves))
        for title, suffix in metrics.items():
            relevant_cols = [c for c in df.columns if c.endswith(suffix)]
            if not relevant_cols: continue

            plt.figure(figsize=(12, 7))
            x_labels = [c.replace(suffix, '') for c in relevant_cols]
            
            for i, (_, row) in enumerate(plot_df.iterrows()):
                label = row[label_col]
                values = row[relevant_cols].values
                plt.plot(x_labels, values, marker='o', label=label, color=colors[i])

            titlef = f'Summary of {title} Metrics ↑' if title in ['Dice', 'Jaccard', 'BIoU', 'OCA'] else f'Summary of {title} Metrics ↓' 

            plt.title(titlef)
            plt.xlabel('Ground Truth vs Model')
            plt.ylabel('Mean')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.6)
            if not is_single or len(plot_df) > 1:
                plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            save_name = folder_path / f"summary_plot_{title.lower()}.png"
            plt.savefig(save_name,dpi=300)
            plt.close()


def process_single_folder(folder_path: Path):
    mean_data = []
    std_data = []
    var_data = []
    
    output_filenames = [
        'mean_summary.csv', 
        'std_summary.csv', 
        'variance_summary.csv', 
        'min_max_summary.csv',
        'single_summary.csv' 
    ]
  
    csv_files = list(folder_path.glob('*.csv'))
    input_csv_files = [f for f in csv_files if f.name not in output_filenames]
    
    if not input_csv_files:
        return

    for file in tqdm(input_csv_files, desc=f"Summary process ({folder_path.name})"):
        try:
            file_id = file.stem.split('_')[-1]
            df = pd.read_csv(file)

            if 'Model' not in df.columns:
                continue

            model_index = df.columns.get_loc('Model')
            target_cols = df.columns[model_index + 1:]
            filtered_df = df[target_cols]

            mean_series = filtered_df.mean(numeric_only=True)
            std_series = filtered_df.std(numeric_only=True)
            var_series = filtered_df.var(numeric_only=True)

            mean_dict = mean_series.to_dict()
            std_dict = std_series.to_dict()
            var_dict = var_series.to_dict()

            mean_dict['id'] = file_id
            std_dict['id'] = file_id
            var_dict['id'] = file_id

            mean_data.append(mean_dict)
            std_data.append(std_dict)
            var_data.append(var_dict)

        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    processed_count = len(mean_data)
    if processed_count == 0:
        return

    if processed_count == 1:        
        mean_result = mean_data[0]
        std_result = std_data[0]
        var_result = var_data[0]
        
        mean_result.pop('id', None) 
        std_result.pop('id', None)
        var_result.pop('id', None)
        
        mean_result['Statistic'] = 'mean'
        std_result['Statistic'] = 'std'
        var_result['Statistic'] = 'variance'
        
        single_summary_list = [mean_result, std_result, var_result]
        df_single = pd.DataFrame(single_summary_list)
        
        cols = ['Statistic'] + [c for c in df_single.columns if c != 'Statistic']
        df_single = df_single[cols]
        
        output_path = folder_path / 'single_summary.csv' 
        df_single.to_csv(output_path, index=False)
        save_summary_plots(df_single, folder_path, is_single=True)
        return 

    def save_summary(data_list, filename):
        if not data_list: return None
        summary_df = pd.DataFrame(data_list)
        cols = ['id'] + [c for c in summary_df.columns if c != 'id']
        summary_df = summary_df[cols]
        output_path = folder_path / filename
        summary_df.to_csv(output_path, index=False)
        return summary_df

    df_mean_all = save_summary(mean_data, 'mean_summary.csv')
    if df_mean_all is not None:
        save_summary_plots(df_mean_all, folder_path, is_single=False)
    df_std_all = save_summary(std_data, 'std_summary.csv')
    df_var_all = save_summary(var_data, 'variance_summary.csv')

    min_max_rows = []
    def extract_min_max_ids(df, metric_name):
        if df is None or df.empty: return
        df_indexed = df.set_index('id')
        row_min = {'Statistic': f'min_{metric_name}'}
        row_max = {'Statistic': f'max_{metric_name}'}
        
        for col in df_indexed.columns:
            min_val = df_indexed[col].min()
            max_val = df_indexed[col].max()

            if min_val == max_val:
                row_min[col] = 'independent of model election'
                row_max[col] = 'independent of model election'  
            else:
                row_min[col] = ",".join(map(str, df_indexed.index[df_indexed[col] == min_val].tolist()))
                row_max[col] = ",".join(map(str, df_indexed.index[df_indexed[col] == max_val].tolist()))

        min_max_rows.extend([row_min, row_max])

    if df_mean_all is not None: extract_min_max_ids(df_mean_all, 'mean')
    if df_std_all is not None: extract_min_max_ids(df_std_all, 'std')
    if df_var_all is not None: extract_min_max_ids(df_var_all, 'variance')

    if min_max_rows:
        df_min_max = pd.DataFrame(min_max_rows)
        cols = ['Statistic'] + [c for c in df_min_max.columns if c != 'Statistic']
        df_min_max = df_min_max[cols]
        final_path = folder_path / 'min_max_summary.csv'
        df_min_max.to_csv(final_path, index=False)


def generate_statistical_summaries(folder_path: Path):
    output_filenames = [
        'mean_summary.csv', 
        'std_summary.csv', 
        'variance_summary.csv', 
        'min_max_summary.csv',
        'single_summary.csv' 
    ]
    
    root_csvs = [f for f in folder_path.glob('*.csv') if f.name not in output_filenames]
    if root_csvs:
        process_single_folder(folder_path)
    else:
        subdirs = [d for d in folder_path.iterdir() if d.is_dir()]
        for subdir in subdirs:
            subdir_csvs = [f for f in subdir.glob('*.csv') if f.name not in output_filenames]
            if subdir_csvs:
                process_single_folder(subdir)


##################################################### single plot functions ################################################################
def plot_curves(dataframe, columns, title, filename, color_list_or_map, y_label):
    plt.figure(figsize=(12, 7))

    legend_handles = []
    legend_labels = []

    colors_to_use = color_list_or_map if isinstance(color_list_or_map, list) else [color_list_or_map(i) for i in range(len(columns))]

    for i, col in enumerate(columns):
        color = colors_to_use[i % len(colors_to_use)]
        line, = plt.plot(dataframe.index, dataframe[col], label=col, color=color, linewidth=1.5)

    plt.title(f'{title}', fontsize=16)
    plt.xlabel('Images', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    plt.legend(handles=legend_handles, labels=legend_labels, title="Columns", 
               bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.grid(False) 
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_distributions(dataframe, columns, title, filename_base, y_label):
    data_to_plot = [dataframe[col].dropna().values for col in columns]
    column_labels = [col.replace('_jaccard', '').replace('_dice', '') for col in columns]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=column_labels, patch_artist=True)
    plt.title(title, fontsize=16)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(False)
            
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f'{filename_base}_boxplot.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.violinplot(data_to_plot, showmeans=True, showmedians=False, showextrema=True)
    plt.xticks(np.arange(1, len(column_labels) + 1), column_labels, ha='right')
    plt.title(title, fontsize=16)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(False)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f'{filename_base}_violinplot.png', dpi=300)
    plt.close()


def graph_generator(input_path):
    df = pd.read_csv(input_path)
    eval_id = str(input_path.name).replace('model_evaluation_','').replace('.csv','')
    output_path = input_path.parent
    (output_path / (eval_id +'_graphs')).mkdir(parents=True, exist_ok=True) 

    counts_ann = ['Ground_Truth', 'Model']
    jaccard_ann = [c for c in df.columns if c.endswith('_jaccard')]
    dice_ann = [c for c in df.columns if c.endswith('_dice')]
    hausdorff_ann = [c for c in df.columns if c.endswith('_hausdorff')]
    msd_ann = [c for c in df.columns if c.endswith('_msd')]
    biou_ann = [c for c in df.columns if c.endswith('_biou')]
    oca_ann = [c for c in df.columns if c.endswith('_oca')]

    colors = plt.cm.get_cmap('tab10', max(len(jaccard_ann), len(dice_ann)))

    plot_curves(df, counts_ann, 'Ground Truth vs Model Findings', output_path / (eval_id +'_graphs') / 'gt_model_findings.png', ['blue', 'green'], 'Findings')
    plot_curves(df, jaccard_ann, 'Ground Truth vs Model', output_path / (eval_id +'_graphs') / 'gt_model_jaccard.png', colors, 'Jaccard index')
    plot_curves(df, dice_ann, 'Ground Truth vs Model', output_path / (eval_id +'_graphs') / 'gt_model_dice.png', colors, 'Dice index')
    plot_curves(df, hausdorff_ann, 'Ground Truth vs Model', output_path / (eval_id +'_graphs') / 'gt_model_hausdorff.png', colors, 'Hausdorff distance')
    plot_curves(df, msd_ann, 'Ground Truth vs Model', output_path / (eval_id +'_graphs') / 'gt_model_msd.png', colors, 'Mean surface distance')
    plot_curves(df, biou_ann, 'Ground Truth vs Model', output_path / (eval_id +'_graphs') / 'gt_model_biou.png', colors, 'Boundary IoU')
    plot_curves(df, oca_ann, 'Ground Truth vs Model', output_path / (eval_id +'_graphs') / 'gt_model_oca.png', colors, 'Overall Contour Agreement')

    plot_distributions(df, counts_ann, 'Ground Truth vs Model Findings', output_path / (eval_id +'_graphs') / 'gt_model_findings', 'Findings')
    plot_distributions(df, jaccard_ann, 'Ground Truth vs Model', output_path / (eval_id +'_graphs') / 'gt_model_jaccard', 'Jaccard Index')
    plot_distributions(df, dice_ann, 'Ground Truth vs Model', output_path / (eval_id +'_graphs') / 'gt_model_dice', 'Dice Index')
    plot_distributions(df, hausdorff_ann, 'Ground Truth vs Model', output_path / (eval_id +'_graphs') / 'gt_model_hausdorff', 'Hausdorff distance')
    plot_distributions(df, msd_ann, 'Ground Truth vs Model', output_path / (eval_id +'_graphs') / 'gt_model_msd', 'Mean surface distance')
    plot_distributions(df, biou_ann, 'Ground Truth vs Model', output_path / (eval_id +'_graphs') / 'gt_model_biou', 'Boundary IoU')
    plot_distributions(df, oca_ann, 'Ground Truth vs Model', output_path / (eval_id +'_graphs') / 'gt_model_oca', 'Overall Contour Agreement')