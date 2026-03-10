from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
from ipyfilechooser import FileChooser
from tqdm.notebook import tqdm
from core.evaluationFunctions import (
    evaluation_metrics,
    add_missing_files,
    generate_statistical_summaries,
    graph_generator
)

def create_evaluation_menu():
    
    gt_path_widget = FileChooser(
        Path.cwd().as_posix(),
        title='Select grount truth folder',
        select_default=False 
    )
    gt_path_widget.show_only_dirs = True 
    gt_path_widget.layout = widgets.Layout(width='80%')

    yaml_path_widget = FileChooser(
        Path.cwd().as_posix(),
        title='Select yaml training file for multi shape transformations',
        select_default=False 
    )
    yaml_path_widget.filter_pattern = ['*.yaml'] 
    yaml_path_widget.layout = widgets.Layout(width='80%')


    model_dims_widget = widgets.Dropdown(
        options={
            '2D': '2D', 
            '3D': '3D'
        },
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

    outName = widgets.Text(
        value='Evaluation_output',
        placeholder='Folder name',
        description='Output folder name (optional):',
        disabled=False,
        style={'description_width':'initial'},
        layout=widgets.Layout(width='30%')
    )
 

    gt_mode = widgets.Dropdown(
        options={
            'single class': 'single', 
            'multiclass': 'multiclass'
        },
        value='single', 
        description='GT class annotations:',
        disabled=False,
        style={'description_width':'50%'} 
    ) 

    eval_mode = widgets.Dropdown(
        options={
            'single class': 'single', 
            'multiclass to binary': 'all'
        },
        value='single', 
        description='Evaluation class mode:',
        disabled=False,
        style={'description_width':'50%'} 
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

    staple_threshold_widget = widgets.BoundedFloatText(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.01,
        description='STAPLE Threshold:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='200px', display='none') 
    )

    predictions_path_widget.show_only_dirs = True 
    predictions_path_widget.layout = widgets.Layout(width='80%')

    use_dilatation_checkbox = widgets.Checkbox(
        value=False,
        description="Use Dilatation",
        disabled=False
    )

    dilation_pixels_widget = widgets.BoundedIntText(
        value=3,
        min=1,
        step=1,
        description='Radius:',
        disabled=False,
        layout=widgets.Layout(width='150px')
    )

    dilation_mode_widget = widgets.Dropdown(
        options=['disk', 'square'],
        value='disk',
        description='Mode:',
        disabled=False,
        layout=widgets.Layout(width='200px')
    )

    dilation_options_box = widgets.HBox(
        [dilation_pixels_widget, dilation_mode_widget],
        layout=widgets.Layout(margin='0 0 0 20px') 
    )

    verify_images_checkbox = widgets.Checkbox(
        value=True,
        description="Verify missing images.",
        disabled=False,
        ident = False
    )
 

    def on_dilatation_change(change):
        if change['new']:
            dilation_options_box.layout.display = 'flex'
        else:
            dilation_options_box.layout.display = 'none'

    
    def check_yaml(chooser):
        if chooser.selected:
            _,spatial_dim = extract_params(chooser.selected,None)
            if spatial_dim is None:
                model_dims_widget.layout.display = 'flex'
            else:
                model_dims_widget.layout.display = 'none'    


    def check_gt_subfolders(chooser):
        if chooser.selected:
            path = Path(chooser.selected)
            if path.exists():
                subdirs = [d for d in path.iterdir() if d.is_dir()]
                if len(subdirs) > 2:
                    staple_threshold_widget.layout.display = 'flex'
                    save_mask_checkbox.layout.display = 'flex'
                else:
                    staple_threshold_widget.layout.display = 'none'
                    save_mask_checkbox.value = False
                    save_mask_checkbox.layout.display = 'none'
    
    model_dims_widget.layout.display = 'none' 
    use_dilatation_checkbox.observe(on_dilatation_change, names='value')
    dilation_options_box.layout.display = 'flex' if use_dilatation_checkbox.value else 'none'

    gt_path_widget.register_callback(check_gt_subfolders)
    yaml_path_widget.register_callback(check_yaml)

    save_mask_checkbox = widgets.Checkbox(
        value=False,
        description="Save generated masks",
        disabled=False
    )

    run_button = widgets.Button(description="Run Evaluation" , button_style='success', )
    output = widgets.Output()

    display(gt_path_widget, predictions_path_widget,outName,gt_mode,eval_mode,eval_class_widget, staple_threshold_widget, 
            use_dilatation_checkbox, dilation_options_box,verify_images_checkbox, 
            save_mask_checkbox,yaml_path_widget,model_dims_widget, run_button, output)
    
    def on_button_clicked(b):
        with output:
            output.clear_output()
            
            if not gt_path_widget.selected or not predictions_path_widget.selected:
                raise ValueError("Please select correct input folders")
                return
            

            model_dims = None
            if not yaml_path_widget.selected:
                yaml_path = None
            else:
                yaml_path = yaml_path_widget.selected 
                if model_dims_widget.layout.display != 'none':
                    if model_dims_widget.value == '2D':
                        model_dims = 2
                    else:
                        model_dims = 3    


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

            eval_mode_selection = eval_mode.value
            gt_mode_selection = gt_mode.value

            if eval_mode_selection == 'single': 
                eval_class = eval_class_widget.value
            else:
                eval_class = 'all'  
  
            save_mask = save_mask_checkbox.value
            staple_thresh = staple_threshold_widget.value
            folder_name = outName.value
            anotators_files = [item.name for item in selected_gt.iterdir() if item.is_dir()]
            if len(anotators_files) == 0:
                raise ValueError(f"No ground truth folders found in {selected_gt}.")

            dilatation = None
            n_pixels = None
            kernel_shape = None
            if use_dilatation_checkbox.value :
                dilatation = True
                n_pixels = dilation_pixels_widget.value
                kernel_shape = dilation_mode_widget.value

            print(f"###################################### { len(anotators_files) } annotators folders found in GT.######################################")
            
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
                    add_missing_files(files_prediction, anotators_files, selected_gt)

                print('--- Generating evaluation metrics ---')
                evaluation_metrics(
                    anotators_files,
                    current_pred_folder,
                    files_prediction,
                    output_path,
                    selected_gt,
                    save_mask,
                    dilatation,
                    n_pixels,
                    kernel_shape, 
                    eval_class, 
                    staple_thresh,
                    gt_mode_selection,
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

    run_button = widgets.Button(description="Run Sumary" , button_style='success', )
    output = widgets.Output()

    display(path_widget, run_button, output)
    
    def on_button_clicked(b):
        with output:
            output.clear_output()
            selected_path = path_widget.selected

            if not selected_path:
                raise ValueError("Please select correct csv's folders")
                return
            
            selected_path = Path(path_widget.selected)
            
            print('###################################### Sumary generation ############################################')
            generate_statistical_summaries(selected_path)
            print('###################################### Sumary complete ############################################')

    run_button.on_click(on_button_clicked)


def create_single_plots_menu():
    path_widget = FileChooser(
        Path.cwd().as_posix(),
        title= "Select csv's file folder",
        select_default=False 
    )
    path_widget.layout = widgets.Layout(width='80%')

    run_button = widgets.Button(description="Run plot generation" , button_style='success', )
    output = widgets.Output()

    display(path_widget, run_button, output)
    
    def on_button_clicked(b):
        with output:
            output.clear_output()
            selected_path = path_widget.selected

            if not selected_path:
                raise ValueError("Please select correct csv's folders")
                return
            
            selected_path = Path(path_widget.selected)
            files = [f.name for f in selected_path.glob("*model_eval*.csv")]
            print('###################################### Plots generation starting ############################################')
            for file in tqdm(files):
                graph_generator(selected_path/file)
            print('###################################### Plot generation complete ############################################') 
    run_button.on_click(on_button_clicked)
