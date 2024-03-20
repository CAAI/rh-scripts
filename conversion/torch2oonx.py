import torch
import torchio as tio
from torch.utils.data import DataLoader
from torchio.data import GridSampler, GridAggregator
from rhtorch.utilities.config import UserConfig
from rhtorch.utilities.modules import (
    recursive_find_python_class,
    find_best_checkpoint
)
import numpy as np
from pathlib import Path
import argparse
import nibabel as nib
import sys
from tqdm import tqdm


def infer_data_from_model(model, subject, ps=None, po=None, bs=1, GPU=True):
    """Infer a full volume given a trained model for 1 patient

    Args:
        model (torch.nn.Module): trained pytorch model
        subject (torchio.Subject): Subject instance from TorchIO library
        ps (list, optional): Patch size (from config). Defaults to None.
        po (int or list, optional): Patch overlap. Defaults to None.
        bs (int, optional): batch_size (from_config). Defaults to 1.

    Returns:
        [np.ndarray]: Full volume inferred from model
    """
    grid_sampler = GridSampler(subject, ps, po)
    patch_loader = DataLoader(grid_sampler, batch_size=bs)
    aggregator = GridAggregator(grid_sampler, overlap_mode='average')
    with torch.no_grad():
        for patches_batch in patch_loader:
            patch_x, _ = model.prepare_batch(patches_batch)
            if GPU:
                patch_x = patch_x.to('cuda')
            locations = patches_batch[tio.LOCATION]
            patch_y = model(patch_x)
            aggregator.add_batch(patch_y, locations)
    return aggregator.get_output_tensor()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Infer new data from input model.')
    parser.add_argument("-c", "--config",
                        help="Config file of saved model",
                        type=str, default='config.yaml')
    parser.add_argument("--checkpoint",
                        help="Choose specific checkpoint that overwrites the config",
                        type=str, default=None)
    parser.add_argument("-o", "--onnx",
                        help="Output onnx path",
                        type=str, default='model.onnx')

    args = parser.parse_args()

    # load configs in inference mode
    user_configs = UserConfig(args, mode='infer')
    model_dir = user_configs.rootdir
    configs = user_configs.hparams
    project_dir = Path(configs['project_dir'])
    model_name = configs['model_name']
    data_shape_in = configs['data_shape_in']
    patch_size = configs['patch_size']
    channels_in = data_shape_in[0]
    
    input_sample = torch.randn([1,channels_in,]+patch_size)

    # load the model
    module_name = recursive_find_python_class(configs['module'])
    model = module_name(configs, data_shape_in)
    
    if args.checkpoint is None:
        # Load the final (best) model
        if 'best_model' in configs:
            ckpt_path = Path(configs['best_model'])
            epoch_suffix = ''
        # Not done training. Load the most recent (best) ckpt
        else:
            ckpt_path = find_best_checkpoint(project_dir.joinpath('trained_models', model_name, 'checkpoints'))
            epoch_suffix = None
    else:
        ckpt_path = args.checkpoint
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    
    # Export the model
    torch.onnx.export(model,                   # model being run
                    input_sample,              # model input (or a tuple for multiple inputs)
                    args.onnx,                 # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                  'output' : {0 : 'batch_size'}})