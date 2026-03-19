import os
# Suppress specific numpy DeprecationWarning
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:numpy"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='monai')
warnings.filterwarnings("ignore", category=UserWarning, message='^In the future `np.bool` will be defined as the corresponding NumPy scalar')
warnings.filterwarnings("ignore", category=DeprecationWarning, message='^In the future `np.bool` will be defined as the corresponding NumPy scalar')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='numpy')
warnings.filterwarnings("ignore",category=FutureWarning)

import copy
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
from mmv_im2im.configs.config_base import ProgramConfig, configuration_validation, parse_adaptor
from mmv_im2im.proj_tester import ProjectTester
import numpy as np
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
from mmv_im2im.utils.utils import topology_preserving_thinning
from dataclasses import dataclass
from pyrallis import field
import argparse
import dataclasses
import sys
from argparse import HelpFormatter, Namespace
from collections import defaultdict
from logging import getLogger
from typing import Dict, List, Sequence, Text, Type, Union, TypeVar, Generic, Optional
from pyrallis import utils, cfgparsing
from pyrallis.help_formatter import SimpleHelpFormatter
from pyrallis.parsers import decoding
from pyrallis.utils import Dataclass, PyrallisException
from pyrallis.wrappers import DataclassWrapper
from tqdm.notebook import tqdm
import gc
import yaml
import torch
import bioio_tifffile
from ipyfilechooser import FileChooser

logger = getLogger(__name__)
T = TypeVar("T")

class ArgumentParser(Generic[T], argparse.ArgumentParser):
    def __init__(
        self,
        config_class: Type[T],
        config: Optional[str] = None,
        formatter_class: Type[HelpFormatter] = SimpleHelpFormatter,
        *args,
        **kwargs,
    ):
        kwargs["formatter_class"] = formatter_class
        super().__init__(*args, **kwargs)

        self.constructor_arguments: Dict[str, Dict] = defaultdict(dict)

        self._wrappers: List[DataclassWrapper] = []

        self.config = config
        self.config_class = config_class

        self._assert_no_conflicts()
        self.add_argument(
            f"--{utils.CONFIG_ARG}",
            type=str,
            help="Path for a config file to parse with pyrallis",
        )
        self.set_dataclass(config_class)

    def set_dataclass(
        self,
        dataclass: Union[Type[Dataclass], Dataclass],
        prefix: str = "",
        default: Union[Dataclass, Dict] = None,
        dataclass_wrapper_class: Type[DataclassWrapper] = DataclassWrapper,
    ):
        if not isinstance(dataclass, type):
            default = dataclass if default is None else default
            dataclass = type(dataclass)

        new_wrapper = dataclass_wrapper_class(dataclass, prefix=prefix, default=default)
        self._wrappers.append(new_wrapper)
        self._wrappers += new_wrapper.descendants

        for wrapper in self._wrappers:
            logger.debug(
                f"Adding arguments for dataclass: {wrapper.dataclass} "
                f"at destination {wrapper.dest}"
            )
            wrapper.add_arguments(parser=self)

    def _assert_no_conflicts(self):
        if utils.CONFIG_ARG in [
            field.name for field in dataclasses.fields(self.config_class)
        ]:
            raise PyrallisException(
                f"{utils.CONFIG_ARG} is a reserved word for pyrallis"
            )

    def parse_args(self, args=None, namespace=None) -> T:
        return super().parse_args(args, namespace)

    def parse_known_args(
        self,
        args: Sequence[Text] = None,
        namespace: Namespace = None,
        attempt_to_reorder: bool = False,
    ):
        if args is None:
            args = sys.argv[1:]
        else:
            args = list(args)

        if "--help" not in args:
            for action in self._actions:
                action.default = argparse.SUPPRESS
                action.type = str
        parsed_args, unparsed_args = super().parse_known_args(args, namespace)

        parsed_args = self._postprocessing(parsed_args)
        return parsed_args, unparsed_args

    def print_help(self, file=None):
        return super().print_help(file)

    def _postprocessing(self, parsed_args: Namespace) -> T:
        logger.debug("\nPOST PROCESSING\n")
        logger.debug(f"(raw) parsed args: {parsed_args}")

        parsed_arg_values = vars(parsed_args)

        for key in parsed_arg_values:
            parsed_arg_values[key] = cfgparsing.parse_string(parsed_arg_values[key])

        config = self.config

        if utils.CONFIG_ARG in parsed_arg_values:
            new_config = parsed_arg_values[utils.CONFIG_ARG]
            if config is not None:
                warnings.warn(
                    UserWarning(f"Overriding default {config} with {new_config}")
                )
            if Path(new_config).is_file():
                config = new_config
            else:
                new_config = str(new_config)
                print(f"trying to locate preset config for {new_config} ...")

                config = Path(__file__).parent / f"preset_{new_config}.yaml"
            del parsed_arg_values[utils.CONFIG_ARG]

        if config is not None:
            print(f"loading configuration from {config} ...")
            file_args = cfgparsing.load_config(open(config, "r"))
            file_args = utils.flatten(file_args, sep=".")
            file_args.update(parsed_arg_values)
            parsed_arg_values = file_args
            print("configuration loading is completed")

        deflat_d = utils.deflatten(parsed_arg_values, sep=".")
        cfg = decoding.decode(self.config_class, deflat_d)

        return cfg

def parse_adaptor_jpnb(
    config_class: Type[T],
    config: Optional[Union[Path, str]] = None,
    args: Optional[Sequence[str]] = None,
) -> T:
    parser = ArgumentParser(config_class=config_class, config=config)
    return parser.parse_args(args=[])

def create_inference_menu():
    
    # 1. Pipeline Mode Selection
    pipeline_mode_header = widgets.Label(
        value="--- Pipeline mode ---",
        style={'font_weight': 'bold'}
    )
    
    pipeline_mode_dropdown = widgets.Dropdown(
        options=['single model', 'multi model'],
        value='single model',
        description='Mode:',
        disabled=False,
        style={'description_width': 'initial'}
    )

    # 2. Config Options Header
    config_header = widgets.Label(
        value="--- Config Options ---",
        style={'font_weight': 'bold'}
    )

    # YAML config file path
    yaml_path_widget = FileChooser(
        Path.cwd().as_posix(),
        title='Select the YAML configuration file',
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

    # --- SINGLE MODEL WIDGETS ---
    ckpt_path_widget = FileChooser(
        Path.cwd().as_posix(),
        title='Select the CKPT model weights file',
        select_default=False 
    )
    ckpt_path_widget.filter_pattern = ['*.ckpt']
    ckpt_path_widget.layout = widgets.Layout(width='80%')

    # --- MULTI MODEL WIDGETS ---
    models_folder_widget = FileChooser(
        Path.cwd().as_posix(),
        title='Select the trained models folder',
        select_default=False
    )
    models_folder_widget.show_only_dirs = True
    models_folder_widget.layout = widgets.Layout(width='80%', display='none')

    output_path_widget = FileChooser(
        Path.cwd().as_posix(),
        title='Select predictions output',
        select_default=False
    )
    output_path_widget.show_only_dirs = True
    output_path_widget.layout = widgets.Layout(width='80%', display='none')

    # Sub-options for Multi-Model
    weight_options_dropdown = widgets.Dropdown(
        options=['last', 'min', 'custom'], 
        value='last',
        description='Weight Options:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(display='none', width='50%')
    )

    # Container for dynamic weights (One selector per subfolder)
    custom_weights_container = widgets.VBox(
        layout=widgets.Layout(display='none', margin='10px 0px 10px 20px')
    )
    
    # Input Images
    path_base_widget = FileChooser(
        Path.cwd().as_posix(), 
        title='Select the input prediction images folder',
        select_default=False 
    )
    path_base_widget.show_only_dirs = True
    path_base_widget.layout = widgets.Layout(width='80%')

   
    def check_yaml(chooser):
        if chooser.selected:
            with open(chooser.selected, "r") as f:
                config = yaml.safe_load(f)
            spatial_dim = config.get("model",{}).get("net",{}).get("params",{}).get("spatial_dims",None)
            if spatial_dim is None:
                model_dims_widget.layout.display = 'flex'
            else:
                model_dims_widget.layout.display = 'none'  
            
    def check_folder_structure():
        if not models_folder_widget.selected:
            return None
        path = Path(models_folder_widget.selected)
        if not path.exists():
            return None
        if list(path.glob('*.ckpt')):
            return "flat"
        subdirs = [x for x in path.iterdir() if x.is_dir()]
        for subdir in subdirs:
            if (subdir / "checkpoints").exists() and (subdir / "checkpoints").is_dir():
                return "nested"
        return "unknown"

    def populate_custom_weight_selectors():
        """New function to create specific dropdowns for each model subfolder"""
        if not models_folder_widget.selected:
            return
        
        models_path = Path(models_folder_widget.selected)
        subdirs = [x for x in models_path.iterdir() if x.is_dir() and (x / "checkpoints").exists()]
        
        # UI headers for the dynamic section
        new_children = [widgets.HTML(value="<b>Select .ckpt for each model:</b>")]
        
        for subdir in subdirs:
            ckpt_dir = subdir / "checkpoints"
            available_ckpts = sorted([f.name for f in ckpt_dir.glob("*.ckpt")])
            
            if not available_ckpts:
                new_children.append(widgets.Label(value=f"⚠️ No weights found in {subdir.name}/checkpoints"))
                continue
                
            # Create a dropdown for this specific folder
            dropdown = widgets.Dropdown(
                options=available_ckpts,
                value='last.ckpt' if 'last.ckpt' in available_ckpts else available_ckpts[0],
                description=f"{subdir.name}:",
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='90%')
            )
            # Store the folder name directly in the widget for easy retrieval later
            dropdown.target_folder = subdir.name
            new_children.append(dropdown)
            
        custom_weights_container.children = tuple(new_children)

    def update_multimodel_ui(change=None):
        structure = check_folder_structure()
        if structure == "nested":
            weight_options_dropdown.layout.display = 'block'
            if weight_options_dropdown.value == 'custom':
                 custom_weights_container.layout.display = 'block'
                 populate_custom_weight_selectors()
            else:
                 custom_weights_container.layout.display = 'none'
        else:
            weight_options_dropdown.layout.display = 'none'
            custom_weights_container.layout.display = 'none'

    def toggle_pipeline_mode(change):
        mode = change['new']
        if mode == 'single model':
            ckpt_path_widget.layout.display = 'block'
            models_folder_widget.layout.display = 'none'
            output_path_widget.layout.display = 'none'
            weight_options_dropdown.layout.display = 'none'
            custom_weights_container.layout.display = 'none'
        else:
            ckpt_path_widget.layout.display = 'none'
            models_folder_widget.layout.display = 'block'
            output_path_widget.layout.display = 'block'
            update_multimodel_ui()
    
    pipeline_mode_dropdown.observe(toggle_pipeline_mode, names='value')
    models_folder_widget.register_callback(update_multimodel_ui)
    weight_options_dropdown.observe(update_multimodel_ui, names='value')

    model_dims_widget.layout.display = 'none' 
    yaml_path_widget.register_callback(check_yaml)

    run_button = widgets.Button(description="Run Inference" , button_style='success', )
    output = widgets.Output()

    # UI Display
    display(
        pipeline_mode_header, pipeline_mode_dropdown,
        config_header,
        yaml_path_widget,
        model_dims_widget, 
        ckpt_path_widget,
        models_folder_widget,
        output_path_widget,
        weight_options_dropdown,
        custom_weights_container,
        path_base_widget, 
        run_button, output
    )

    def parse_int_list(text_value):
        if not text_value.strip():
            return None
        try:
            return [int(x.strip()) for x in text_value.split(',') if x.strip()]
        except ValueError:
            raise ValueError(f"Invalid list: '{text_value}'. Use integers separated by commas.")

    def run_single_inference(cfg_obj, ckpt_path, output_dir_path):
        try:
            print(f"--> Loading Weights: {Path(ckpt_path).name}")
            cfg_obj.model.checkpoint = Path(ckpt_path)
            cfg_obj.data.inference_output.path = Path(output_dir_path)
            cfg_obj.data.inference_output.path.mkdir(parents=True, exist_ok=True)
            
            cfg_obj = configuration_validation(cfg_obj)
            exe = ProjectTester(cfg_obj)
            exe.run_inference()
            
            del exe
            gc.collect()
            torch.cuda.empty_cache()
            print(f"--> Done: {output_dir_path}")
        except Exception as e:
            print(f"Error processing {ckpt_path}: {e}")

    def on_button_clicked(b):
        with output:
            output.clear_output()
            
            mode = pipeline_mode_dropdown.value
            selected_yaml = yaml_path_widget.selected
            selected_path_base = Path(path_base_widget.selected)            
            if not selected_yaml or not selected_path_base:
                print("Error: Required paths missing.")
                return

            try:
                base_cfg = parse_adaptor_jpnb(config_class=ProgramConfig, config=selected_yaml)
                
                if 'spatial_dims' in  base_cfg.model.net['params'] : 
                    spatial_dims = base_cfg.model.net.get('params', {}).get('spatial_dims', 2)
                else:
                    spatial_dims = None
                    if model_dims_widget.layout.display != 'none':
                        if model_dims_widget.value == '2D':
                            spatial_dims = 2
                        else:
                            spatial_dims = 3
                
                if model_dims_widget.layout.display != 'none' and 'ProbUnet' in base_cfg.model.framework:
                    base_cfg.model.framework = base_cfg.model.framework + '_old'
                    base_cfg.model.net['module_name'] =  base_cfg.model.net['module_name']+'_old'
                
                # Set configs
                base_cfg.mode = 'inference'
                base_cfg.data.inference_input.dir = Path(selected_path_base) 
            except Exception as e:
                print(f"Config Error: {e}")
                return

            if mode == 'single model':
                if not ckpt_path_widget.selected:
                    print("Error: Select a CKPT file.")
                    return
                run_single_inference(copy.deepcopy(base_cfg), ckpt_path_widget.selected, Path(selected_path_base).parent / "model_predictions")
                print("######################## Prediction Ready #############################")

            elif mode == 'multi model':
                models_path = Path(models_folder_widget.selected)
                output_path = Path(output_path_widget.selected)
                structure = check_folder_structure() 
                inference_tasks = [] 

                if structure == "flat":
                    for f in models_path.glob("*.ckpt"):
                        inference_tasks.append((f, f"predictions_{f.stem}"))
                
                elif structure == "nested":
                    weight_opt = weight_options_dropdown.value
                    if weight_opt == "last":
                        for subdir in [x for x in models_path.iterdir() if x.is_dir()]:
                            ckpt = subdir / "checkpoints" / "last.ckpt"
                            if ckpt.exists(): inference_tasks.append((ckpt, f"predictions_{subdir.name}"))
                    elif weight_opt == "custom":
                        # Retrieve specific ckpt for each folder from the dynamic dropdowns
                        for widget in custom_weights_container.children:
                            if isinstance(widget, widgets.Dropdown):
                                ckpt_name = widget.value
                                folder = widget.target_folder
                                ckpt_path = models_path / folder / "checkpoints" / ckpt_name
                                if ckpt_path.exists():
                                    inference_tasks.append((ckpt_path, f"predictions_{folder}"))
                    elif weight_opt == "min":
                        for subdir in [x for x in models_path.iterdir() if x.is_dir()]:
                            ckpt_dir = subdir / "checkpoints"
                            if not ckpt_dir.exists(): continue
                            
                            best_ckpt = None
                            min_val_loss = float('inf')
                            
                            # Iterate over all ckpt files to find the one with lowest val_loss
                            for f in ckpt_dir.glob("*.ckpt"):
                                # Expecting format: epoch=int-val_loss=float.ckpt
                                if "val_loss=" in f.name:
                                    try:
                                        # Extract the number part after val_loss=
                                        loss_str = f.name.split("val_loss=")[1]
                                        # Remove .ckpt extension to parse the float
                                        loss_str = loss_str.replace(".ckpt", "")
                                        val_loss = float(loss_str)
                                        
                                        if val_loss < min_val_loss:
                                            min_val_loss = val_loss
                                            best_ckpt = f
                                    except ValueError:
                                        # Skip files that don't match the expected float format
                                        continue
                            
                            if best_ckpt:
                                inference_tasks.append((best_ckpt, f"predictions_{subdir.name}"))
                            else:
                                print(f"Warning: No valid 'val_loss' checkpoint found in {subdir.name}")

                if not inference_tasks:
                    print("No valid models found.")
                    return

                for i, (ckpt, out_name) in enumerate(inference_tasks):
                    output.clear_output(wait=True)
                    print(f"\nProcessing {i+1}/{len(inference_tasks)}: {out_name}")
                    run_single_inference(copy.deepcopy(base_cfg), ckpt, output_path / out_name)
                print(f"######################## Predictios for {len(inference_tasks)} models done #############################")
                print("\n######################## All Predictions Ready #############################")


    run_button.on_click(on_button_clicked)

