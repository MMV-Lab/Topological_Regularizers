import os
# Suppress specific numpy DeprecationWarning
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:numpy"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='monai')
warnings.filterwarnings("ignore", category=UserWarning, message='^In the future `np.bool` will be defined as the corresponding NumPy scalar')
warnings.filterwarnings("ignore", category=DeprecationWarning, message='^In the future `np.bool` will be defined as the corresponding NumPy scalar')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='numpy')
warnings.filterwarnings("ignore", category=FutureWarning)
import copy
from pathlib import Path
import argparse
from argparse import HelpFormatter, Namespace
from collections import defaultdict
from dataclasses import dataclass
import dataclasses
import sys
from logging import getLogger
from typing import Dict, List, Sequence, Text, Type, Union, TypeVar, Generic, Optional
from pyrallis import utils, cfgparsing
from pyrallis.help_formatter import SimpleHelpFormatter
from pyrallis.parsers import decoding
from pyrallis.utils import Dataclass, PyrallisException
from pyrallis.wrappers import DataclassWrapper
from tqdm import tqdm
import gc
import torch
from mmv_im2im.configs.config_base import ProgramConfig, configuration_validation
from mmv_im2im.map_extractor import MapExtractor

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
        """Creates an ArgumentParser instance."""
        kwargs["formatter_class"] = formatter_class
        super().__init__(*args, **kwargs)

        # constructor arguments for the dataclass instances.
        # (a Dict[dest, [attribute, value]])
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
        """Adds command-line arguments for the fields of `dataclass`."""
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
        """Checks for a field name that conflicts with utils.CONFIG_ARG"""
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
        # NOTE: since the usual ArgumentParser.parse_args() calls
        # parse_known_args, we therefore just need to overload the
        # parse_known_args method to support both.
        if args is None:
            # args default to the system args
            args = sys.argv[1:]
        else:
            # make sure that args are mutable
            args = list(args)

        if "--help" not in args:
            for action in self._actions:
                # TODO: Find a better way to do that?
                action.default = (
                    argparse.SUPPRESS
                )  # To avoid setting of defaults in actual run
                action.type = (
                    str  # In practice, we want all processing to happen with yaml
                )
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

        config = self.config  # Could be NONE

        if utils.CONFIG_ARG in parsed_arg_values:
            new_config = parsed_arg_values[utils.CONFIG_ARG]
            if config is not None:
                warnings.warn(
                    UserWarning(f"Overriding default {config} with {new_config}")
                )
            ######################################################################
            # adapted from original implementation in pyrallis
            ######################################################################
            if Path(new_config).is_file():
                # pass in a absolute path
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


def parse_adaptor(
    config_class: Type[T],
    config: Optional[Union[Path, str]] = None,
    args: Optional[Sequence[str]] = None,
) -> T:
    parser = ArgumentParser(config_class=config_class, config=config)
    return parser.parse_args(args)

def check_folder_structure(models_path: Path):
    if not models_path.exists():
        return None
    if list(models_path.glob('*.ckpt')):
        return "flat"
    subdirs = [x for x in models_path.iterdir() if x.is_dir()]
    for subdir in subdirs:
        if (subdir / "checkpoints").exists() and (subdir / "checkpoints").is_dir():
            return "nested"
    return "unknown"

def run_single_inference(cfg_obj, ckpt_path, output_dir_path):
    try:
        print(f"--> Loading Weights: {Path(ckpt_path).name}")
        cfg_obj.model.checkpoint = Path(ckpt_path)
        cfg_obj.data.inference_output.path = Path(output_dir_path)
        cfg_obj.data.inference_output.path.mkdir(parents=True, exist_ok=True)
        
        cfg_obj = configuration_validation(cfg_obj)
        exe = MapExtractor(cfg_obj)
        exe.run_inference()
        
        del exe
        gc.collect()
        torch.cuda.empty_cache()
        print(f"--> Done: {output_dir_path}")
    except Exception as e:
        print(f"Error processing {ckpt_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="MMV Im2Im Inference CLI - YAML Driven", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Core Requirement
    parser.add_argument('--images_folder', type=str, help='Path to the input images')
    parser.add_argument('--yaml_path', type=str, required=True, help='Path to the fully configured YAML file')
    
    # Pipeline Execution Mode
    parser.add_argument('--pipeline_mode', type=str, choices=['single', 'multi'], default='single', help='Pipeline execution mode')
    
    # Multi-Model Specific Arguments (Only required if pipeline_mode == 'multi')
    parser.add_argument('--models_folder', type=str, help='Path to the base folder containing multiple training models')
    parser.add_argument('--multi_output_dir', type=str, help='Base directory to save predictions for multiple models')
    parser.add_argument('--weight_option', type=str, choices=['last', 'min', 'custom'], default='last', help='Weight selection strategy for nested folders')
    parser.add_argument('--custom_weights', type=str, nargs='+', help='Custom weights format: "FolderA:best.ckpt" "FolderB:epoch=1.ckpt"')

    args = parser.parse_args()

    # Validate multi-mode requirements
    if args.pipeline_mode == 'multi':
        if not args.models_folder or not args.multi_output_dir:
            parser.error("--models_folder and --multi_output_dir are required when --pipeline_mode is 'multi'")

    print(f"Loading YAML config: {args.yaml_path} ...")
    try:
        base_cfg = parse_adaptor(config_class=ProgramConfig, config=args.yaml_path, args=[])
    except Exception as e:
        print(f"Failed to load YAML configuration: {e}")
        sys.exit(1)
    
    # Automatically handle dynamic parameter injection based on YAML contents
    spatial_dims = base_cfg.model.net.get('params', {}).get('spatial_dims', 2)

    if 'ProbUnet' in base_cfg.model.framework:
        base_cfg.model.framework = base_cfg.model.framework + '_old'
        base_cfg.model.net['module_name'] = base_cfg.model.net['module_name'] + '_old'

    if spatial_dims == 3:
        inference_mode = 'vol2vol'
        print(f"Info: Spatial dimensions = 3. Using volumetric inference mode ({inference_mode}).")
    else:
        inference_mode = 'vol2slice'
        print(f"Info: Spatial dimensions = 2. Using slice-based inference mode ({inference_mode}).")

    if 'pred_slice2vol' not in base_cfg.model.net:
        base_cfg.model.net['pred_slice2vol'] = {}

    yaml_max_proj = base_cfg.model.net['pred_slice2vol'].get('max_proj', False)
    if inference_mode == 'vol2vol' and yaml_max_proj:
        base_cfg.model.net['pred_slice2vol']['max_proj'] = False
        print("Info: Max projection disabled due to vol2vol inference mode.")

    base_cfg.model.net['pred_slice2vol']['jupyter'] = False
    base_cfg.model.net['pred_slice2vol']['inference_mode'] = inference_mode

    if '_old' in base_cfg.model.framework:
        base_cfg.model.net['pred_slice2vol']['n_class_correction'] = base_cfg.model.net.get('params', {}).get('n_classes', 1)
    else:
        base_cfg.model.net['pred_slice2vol']['n_class_correction'] = base_cfg.model.net.get('params', {}).get('out_channels', 1)
    base_cfg.data.inference_input.dir = Path(args.images_folder)
    # ---------------------------------------------------------
    # Execution Routing
    # ---------------------------------------------------------
    if args.pipeline_mode == 'single':
        # In single mode, everything (including ckpt_path and output paths) is taken directly from the YAML.
        if not base_cfg.model.checkpoint:
            print("Error: model.checkpoint is not defined in the YAML file.")
            sys.exit(1)
            
        print("######################## Starting Single Inference #############################")
        run_single_inference(copy.deepcopy(base_cfg), base_cfg.model.checkpoint, base_cfg.data.inference_output.path)
        print("######################## Prediction Ready #############################")

    elif args.pipeline_mode == 'multi':
        models_path = Path(args.models_folder)
        output_path = Path(args.multi_output_dir)
        structure = check_folder_structure(models_path)
        inference_tasks = []

        if structure == "flat":
            for f in models_path.glob("*.ckpt"):
                inference_tasks.append((f, f"predictions_{f.stem}"))
        
        elif structure == "nested":
            if args.weight_option == "last":
                for subdir in [x for x in models_path.iterdir() if x.is_dir()]:
                    ckpt = subdir / "checkpoints" / "last.ckpt"
                    if ckpt.exists(): 
                        inference_tasks.append((ckpt, f"predictions_{subdir.name}"))
                        
            elif args.weight_option == "custom":
                if not args.custom_weights:
                    print("Error: --custom_weights list required when using 'custom' weight option.")
                    sys.exit(1)
                custom_map = {}
                for cw in args.custom_weights:
                    if ':' in cw:
                        folder, ckpt_name = cw.split(':', 1)
                        custom_map[folder] = ckpt_name
                
                for folder, ckpt_name in custom_map.items():
                    ckpt_path = models_path / folder / "checkpoints" / ckpt_name
                    if ckpt_path.exists():
                        inference_tasks.append((ckpt_path, f"predictions_{folder}"))
                    else:
                        print(f"Warning: Custom weight {ckpt_path} not found.")
                        
            elif args.weight_option == "min":
                for subdir in [x for x in models_path.iterdir() if x.is_dir()]:
                    ckpt_dir = subdir / "checkpoints"
                    if not ckpt_dir.exists(): continue
                    
                    best_ckpt = None
                    min_val_loss = float('inf')
                    
                    for f in ckpt_dir.glob("*.ckpt"):
                        if "val_loss=" in f.name:
                            try:
                                loss_str = f.name.split("val_loss=")[1].replace(".ckpt", "")
                                val_loss = float(loss_str)
                                
                                if val_loss < min_val_loss:
                                    min_val_loss = val_loss
                                    best_ckpt = f
                            except ValueError:
                                continue
                    
                    if best_ckpt:
                        inference_tasks.append((best_ckpt, f"predictions_{subdir.name}"))
                    else:
                        print(f"Warning: No valid 'val_loss' checkpoint found in {subdir.name}")

        if not inference_tasks:
            print("No valid models or weights found in the specified folder. Exiting.")
            sys.exit(1)

        for i, (ckpt, out_name) in enumerate(inference_tasks):
            print(f"\nProcessing {i+1}/{len(inference_tasks)}: {out_name}")
            run_single_inference(copy.deepcopy(base_cfg), ckpt, output_path / out_name)
            
        print(f"######################## Predictions for {len(inference_tasks)} models done #############################")
        print("\n######################## All Predictions Ready #############################")

if __name__ == "__main__":
    main()