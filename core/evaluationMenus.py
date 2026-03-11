from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
from ipyfilechooser import FileChooser
from tqdm.notebook import tqdm

# Update this import path based on your local directory structure
from core.evaluationFunctions import (
    evaluation_metrics,
    add_missing_files,
    generate_statistical_summaries,
    graph_generator,
    extract_params
)

def create_evaluation_menu():
    
    gt_path_widget = FileChooser(
        Path.cwd().as_posix(),
        title='Select Ground Truth folder',
        select_default=False 
    )
    gt_path_widget.show_only_dirs = True 
    gt_path_widget.layout = widgets.Layout(width='80%')

    yaml_path_widget = FileChooser(
        Path.cwd().as_posix(),
        title='Select yaml training file for multi shape transformations (optional)',
        select_default=False 
    )
    yaml_path_widget.filter_pattern = ['*.yaml'] 
    yaml_path_widget.layout = widgets.Layout(width='80%')

    model_dims_widget = widgets.Dropdown(
        options={'2D': '2D', '3D': '3D'},
        value='2D', 
        description='Spatial model dimension:',
        disabled=False,
        style={'description_width':'50%'} 
    )    

    predictions_path_widget = FileChooser(
        Path.cwd().as_posix(), 
        title='Select model predictions folder',
        select_default=False 
    )
    predictions_path_widget.show_only_dirs = True 
    predictions_path_widget.layout = widgets.Layout(width='80%')    

    outName = widgets.Text(
        value='Evaluation_output',
        placeholder='Folder name',
        description='Output folder name (optional):',
        disabled=False,
        style={'description_width':'initial'},
        layout=widgets.Layout(width='30%')
    )
 
    eval_class_widget = widgets.BoundedIntText(
        value=1,           
        min=0,             
        step=1,
        description='Evaluation Class:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='150px')
    )

    verify_images_checkbox = widgets.Checkbox(
        value=True,
        description="Verify missing images.",
        disabled=False,
        indent=False
    )
 
    def check_yaml(chooser):
        if chooser.selected:
            _, spatial_dim = extract_params(chooser.selected, None)
            if spatial_dim is None:
                model_dims_widget.layout.display = 'flex'
            else:
                model_dims_widget.layout.display = 'none'    

    model_dims_widget.layout.display = 'none' 
    yaml_path_widget.register_callback(check_yaml)

    run_button = widgets.Button(description="Run Evaluation" , button_style='success')
    output = widgets.Output()

    display(gt_path_widget, predictions_path_widget, outName, eval_class_widget, verify_images_checkbox, yaml_path_widget, model_dims_widget, run_button, output)
    
    def on_button_clicked(b):
        with output:
            output.clear_output()
            
            if not gt_path_widget.selected or not predictions_path_widget.selected:
                print("Error: Please select correct input folders")
                return
            
            eval_class = eval_class_widget.value
            model_dims = None
            if not yaml_path_widget.selected:
                yaml_path = None
            else:
                yaml_path = yaml_path_widget.selected 
                if model_dims_widget.layout.display != 'none':
                    model_dims = 2 if model_dims_widget.value == '2D' else 3    

            # Directly load the ground truth folder
            selected_gt = Path(gt_path_widget.selected)
            base_prediction_path = Path(predictions_path_widget.selected)
            
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

            folder_name = outName.value
            
            for current_pred_folder in folders_to_process:
                output.clear_output(wait=True)
                print(f"\n=================================================================================")
                print(f"PROCESSING PREDICTION MODEL: {current_pred_folder.name}")
                print(f"=================================================================================")
                
                files_prediction = sorted(current_pred_folder.glob("*.tiff"))
                files_prediction.extend(list(current_pred_folder.glob('*.tif')))
                
                if not files_prediction:
                    print(f"WARNING: No images found in {current_pred_folder.name}. Skipping.")
                    continue
                
                output_path = current_pred_folder.parent / folder_name
                output_path.mkdir(parents=True, exist_ok=True)
                
                print(f"Found {len(files_prediction)} prediction files.")
                if verify_images_checkbox.value: 
                    print('--- Looking for missing annotations ---')
                    add_missing_files(files_prediction, selected_gt)

                print('--- Generating evaluation metrics ---')
                evaluation_metrics(
                    current_pred_folder,
                    files_prediction,
                    output_path,
                    selected_gt,
                    eval_class, 
                    yaml_path,
                    model_dims
                )
                print(f"Completed: {current_pred_folder.name}")

            print('\n###################################### All evaluations completed ############################################')
            print(f'Evaluation results saved at {base_prediction_path.parent if direct_images else base_prediction_path} / {folder_name}')
            
    run_button.on_click(on_button_clicked)


def create_sumary_menu():
    path_widget = FileChooser(
        Path.cwd().as_posix(),
        title= "Select csv's file folder",
        select_default=False 
    )
    path_widget.layout = widgets.Layout(width='80%')

    run_button = widgets.Button(description="Run Summary" , button_style='success')
    output = widgets.Output()

    display(path_widget, run_button, output)
    
    def on_button_clicked(b):
        with output:
            output.clear_output()
            selected_path = path_widget.selected

            if not selected_path:
                print("Error: Please select correct csv's folders")
                return
            
            selected_path = Path(path_widget.selected)
            
            print('###################################### Summary generation ############################################')
            generate_statistical_summaries(selected_path)
            print('###################################### Summary complete ############################################')

    run_button.on_click(on_button_clicked)


def create_single_plots_menu():
    path_widget = FileChooser(
        Path.cwd().as_posix(),
        title= "Select csv's file folder",
        select_default=False 
    )
    path_widget.layout = widgets.Layout(width='80%')

    run_button = widgets.Button(description="Run plot generation" , button_style='success')
    output = widgets.Output()

    display(path_widget, run_button, output)
    
    def on_button_clicked(b):
        with output:
            output.clear_output()
            selected_path = path_widget.selected

            if not selected_path:
                print("Error: Please select correct csv's folders")
                return
            
            selected_path = Path(path_widget.selected)
            files = [f.name for f in selected_path.glob("*model_eval*.csv")]
            print('###################################### Plots generation starting ############################################')
            for file in tqdm(files):
                graph_generator(selected_path/file)
            print('###################################### Plot generation complete ############################################') 
            
    run_button.on_click(on_button_clicked)