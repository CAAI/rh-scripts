from nipype.interfaces.fsl import (
    Threshold,
    IsotropicSmooth,
    RobustFOV,
    BET,
    FLIRT,
    ConvertXFM,
    Reorient2Std
)
from nipype.interfaces.fsl.maths import ApplyMask, BinaryMaths, UnaryMaths
from nipype.interfaces.niftyreg import RegResample, RegAladin, RegTransform
from nipype.interfaces.freesurfer import RobustRegister, ApplyVolTransform
import typing
import pathlib
import os
import nibabel as nib


def load_nii(path, target_orientation=('R','A','S'), reorient=True, return_image_only=True):
    
    try:
        img = nib.load(path)
    
        if reorient and not nib.aff2axcodes(img.affine) == target_orientation:
            ornt_orig = nib.orientations.io_orientation(img.affine)
            ornt_targ = nib.orientations.axcodes2ornt(''.join(target_orientation))
            transform = nib.orientations.ornt_transform(ornt_orig, ornt_targ)
    
            img = img.as_reoriented(transform)
    
        if return_image_only:
            return img.get_fdata()
        else:
            return img
    except Exception as e:
        print(f"Error loading NIfTI file {path}: {e}")
        return None

######################################################################################################
#################################   Registration utilities   ##########################################
######################################################################################################


def reg_resample(ref_file, flo_file, trans_file, out_file, interpol='NN',
                 pad_val=None, verbosity=None):
    """Resample nifty file to a reference template given a transformation matrix

    Args:
        ref_file (a pathlike object or str): The input reference/target image
        flo_file (a pathlike object or str): The input floating/source image
        trans_file (a pathlike object or str): The input transformation matrix file
        out_file (a pathlike object or str): The output filename of the transformed image
        interpol (‘NN’ or ‘LIN’ or ‘CUB’ or ‘SINC’): Type of interpolation. Defaults to 'NN'.
        pad_val (float, optional): Padding value to pad. Defaults to None.
        verbosity (None or str): One of file, file_split, file_stdout,
                                 file_stderr, stream, allatonce, none

    Returns:
        Runtime object (except for verbosity='none').
        Access errors by e.g.:
            result = reg_resample(...,verbosity='file_stdout')
            result.runtime.stdout
    """

    rsl = RegResample()
    rsl.inputs.ref_file = ref_file
    rsl.inputs.flo_file = flo_file
    rsl.inputs.trans_file = trans_file
    rsl.inputs.inter_val = interpol
    if pad_val is not None:
        rsl.inputs.pad_val = pad_val
    rsl.inputs.out_file = out_file

    if verbosity is not None:
        if verbosity not in ('file', 'file_split', 'file_stdout',
                             'file_stderr', 'stream', 'allatonce', 'none'):
            raise ValueError('Verbosity of a nipype function must be one of '
                             'the specified.')
        rsl.terminal_output = verbosity

    return rsl.run()


def reg_aladin(ref_file, flo_file, aff_file, rig_only_flag=False, in_aff_file=None, res_file=None, verbosity=None):
    """Block Matching algorithm for symmetric global registration

    Args:
        ref_file (a pathlike object or str): The input reference/target image
        flo_file (a pathlike object or str): The input floating/source image
        aff_file (a pathlike object or str): The output affine matrix file
        res_file (a pathlike object or str): The affine transformed floating image
        verbosity (None or str): One of file, file_split, file_stdout,
                                 file_stderr, stream, allatonce, none

    Returns:
        Runtime object (except for verbosity='none').
        Access errors by e.g.:
            result = reg_aladin(...,verbosity='file_stdout')
            result.runtime.stdout
    """

    ral = RegAladin()
    ral.inputs.ref_file = ref_file
    ral.inputs.flo_file = flo_file
    if in_aff_file:
        ral.inputs.in_aff_file = in_aff_file
    if res_file:
        ral.inputs.res_file = res_file
    ral.inputs.aff_file = aff_file
    ral.inputs.rig_only_flag=rig_only_flag
    if verbosity is not None:
        if verbosity not in ('file', 'file_split', 'file_stdout',
                             'file_stderr', 'stream', 'allatonce', 'none'):
            raise ValueError('Verbosity of a nipype function must be one of '
                             'the specified.')
        ral.terminal_output = verbosity
    return ral.run()


def inv_affine(inv_aff_input, out_file):
    """Invert an affine transformation file

    Args:
        inv_aff_input (a pathlike object or str): The input affine transfrom file
        out_file (a pathlike object or str): The output filename
    """

    inverter = RegTransform()
    inverter.inputs.inv_aff_input = inv_aff_input
    inverter.inputs.out_file = out_file
    inverter.run()


def iso_resample(in_file, out_file, voxel_size=1.0):
    """Resamples an existing volume with the target voxel size

    Args:
        in_file (a pathlike object or str): Input file
        out_file (a pathlike object or str): Output file
        voxel_size (float): target voxel sixe in mm. Default 1.0
    """
    flirt( in_file=in_file,
           ref_file=in_file,
           out_file=out_file,
           output_type='NIFTI_GZ',
           apply_isoxfm=voxel_size
          )

def flirt(in_file, ref_file, out_file, **kwargs):
    """Perform resampling or registration using FSL FLIRT

    Args:
        in_file (a pathlike object or str): Input file to be moved
        ref_file (a pathlike object or str): Target or reference file
        out_file (a pathlike object or str): Output file
        kwargs: Arguments to FLIRT, for options see:
                https://nipype.readthedocs.io/en/0.12.1/interfaces/generated/nipype.interfaces.fsl.preprocess.html#flirt

    Example usage:
        from rhscripts.nifty import flirt
        flirt('MRI.nii.gz', 'MNI.nii.gz', 'MRI_rsl.nii.gz', dof=6, interp='spline')
    """
    resampler = FLIRT()
    resampler.inputs.in_file = in_file
    resampler.inputs.reference = ref_file
    resampler.inputs.out_file = out_file
    for key,value in kwargs.items():
        if hasattr( resampler.inputs, key ):
            setattr( resampler.inputs, key, value )
    try:
        resampler.run()
    except Exception as e:
        print(e)


######################################################################################################
#################################   FSL wrapper utilities   ##########################################
######################################################################################################


def apply_mask(in_file, mask_file, out_file):
    """Use fslmaths to apply a binary mask to another image

    Args:
        in_file (a pathlike object or str): Image to operate on
        mask_file (a pathlike object or str): Binary image defining mask space
        out_file (a pathlike object or str): Image to write
    """

    mask = ApplyMask()
    mask.inputs.in_file = in_file
    mask.inputs.mask_file = mask_file
    mask.inputs.out_file = out_file
    mask.run()


def isotropic_smooth(in_file, out_file, sigma):
    """Gaussian filter

    Args:
        in_file (a pathlike object or str): Input filename.
        out_file (a pathlike object or str): Output filename.
        sigma (float): Gaussian spread

    """

    smoothing = IsotropicSmooth()
    smoothing.inputs.in_file = in_file
    smoothing.inputs.sigma = sigma
    smoothing.inputs.out_file = out_file
    smoothing.run()


def threshold(in_file, out_file, thresh=0.0, direction='below'):
    """Value clipping in an image

    Args:
        in_file (a pathlike object or str): Input filename.
        out_file (a pathlike object or str): Output filename.
        thresh (float, optional): Cutoff value. Defaults to 0.0.
        direction (str, optional): Direction of clipping. Defaults to 'below'.

    """

    clamp = Threshold()
    clamp.inputs.in_file = in_file
    clamp.inputs.thresh = thresh
    clamp.inputs.direction = direction
    clamp.inputs.out_file = out_file
    clamp.run()


def robust_fov(in_file, out_roi, out_transform):
    """Automatically crops an image removing lower head and neck

    Args:
        in_file (a pathlike object or str): Input filename
        out_roi (a pathlike object or str): ROI volume output name
        out_transform (a pathlike object or str): Transformation matrix in_file to out_roi output name
    """

    crop = RobustFOV()
    crop.inputs.in_file = in_file
    crop.inputs.out_roi = out_roi
    crop.inputs.out_transform = out_transform
    crop.run()


def skull_strip(in_file, out_file, frac=0.5, mask=True):
    """Brain extraction routine for skull stripping

    Args:
        in_file (a pathlike object or str): Input filename
        out_file (a pathlike object or str): Output filename
        frac (float): Fractional intensity threshold
        mask (bool, optional): Create binary mask image. Defaults to True.
    """
    bet = BET()
    bet.inputs.in_file = in_file
    bet.inputs.mask = mask
    bet.inputs.frac = frac
    bet.inputs.out_file = out_file
    bet.run()


def hd_bet(input:typing.Union[str,pathlib.Path],
           output:typing.Union[str,pathlib.Path]=None,
           mask:bool=True,
           input_type:str='MR'):
    """GPU-based BET

    Args:
        input (a pathlike object or str): Input filename or folder
                                          When folder, all .nii.gz files will
                                          be processed
        output (a pathlike object or str): Output filename or folder.
                                           When None, output will be
                                           <input>_BET and <input>_BET_mask
        mask (bool, optional): Create binary mask image. Defaults to True.
        input_type (str: optional): Chose MR or CT input.
    """
    if input_type not in ('MR','MRI','CT'):
        raise ValueError('Unknown method')

    if input_type in ('MR','MRI'):
        cmd = 'hd-bet'
    elif input_type == 'CT':
        cmd = 'hd-bet-ct'
        raise ValueError('Method not yet implemented')

    cmd += f' -i {input}'
    if not mask:
        cmd += ' -s 0'
    if output is not None:
        cmd += f' -o {output}'
    os.system(cmd)


def rescale(in_file, out_file, operand_value):
    """Multiplies the full volume with a numeric value

    Args:
        in_file (a pathlike object or str): Image to operate on
        operand_value (float): value to multiply the data
    """

    multiplier = BinaryMaths()
    multiplier.inputs.in_file = in_file
    multiplier.inputs.operation = 'mul'
    multiplier.inputs.operand_value = operand_value
    multiplier.inputs.out_file = out_file
    multiplier.run()


def inv_mask(in_file, out_file):
    """Multiplies the full volume with a numeric value

    Args:
        in_file (a pathlike object or str): Image to operate on
        out_file (a pathlike object or str): Output filename
    """

    inverter = UnaryMaths()
    inverter.inputs.in_file = in_file
    inverter.inputs.operation = 'binv'
    inverter.inputs.out_file = out_file
    inverter.inputs.output_type = "NIFTI_GZ"
    inverter.run()


def merge_images(in_file1, in_file2, out_file):
    """Adds two images together in one file

    Args:
        in_file1 (a pathlike object or str): Input image file 1
        in_file2 (a pathlike object or str): Input image file 2
        out_file (a pathlike object or str): Output filename
    """

    multiplier = BinaryMaths()
    multiplier.inputs.in_file = in_file1
    multiplier.inputs.operation = 'add'
    multiplier.inputs.operand_file = in_file2
    multiplier.inputs.out_file = out_file
    multiplier.run()


def concat_transforms(in_file1, in_file2, out_file=None):
    """Concatenates two affine transforms.

    Args:
        in_file1 (a pathlike object or str): Affine transform file 1
        in_file2 (a pathlike object or str): Affine transform file 2
        out_file (a pathlike object or str, optional): Output filename. Defaults to None (overwrites file 2)
    """

    concat = ConvertXFM()
    concat.inputs.in_file = in_file1
    concat.inputs.in_file2 = in_file2
    concat.inputs.concat_xfm = True
    if not out_file:
        out_file = in_file2   # overwrite file 2
    concat.inputs.out_file = out_file
    concat.run()

def reorient_to_std(in_file: typing.Union[str, pathlib.Path], out_file: typing.Union[str, pathlib.Path]):
    """Reorient orientation to match standard template images (MNI152)

    Args:
        in_file (a pathlike object or str): Input filename
        out_file (a pathlike object or str): Output filename
    """

    reorient = Reorient2Std()
    reorient.inputs.in_file = in_file
    reorient.inputs.out_file = out_file
    res = reorient.run()


######################################################################################################
###############################   FREESURFER REGISTRATION   ##########################################
######################################################################################################

""" Wrapper to mri_robust_register """
def freesurfer_align_volumes_in_common_space(source: typing.Union[str, pathlib.Path], target: typing.Union[str, pathlib.Path], out_lta_file: typing.Union[str, pathlib.Path, None], 
                                             out_half_src_file: typing.Union[str, pathlib.Path, None], out_half_src_lta_file: typing.Union[str, pathlib.Path, None],
                                             out_half_tgt_file: typing.Union[str, pathlib.Path, None], out_half_tgt_lta_file: typing.Union[str, pathlib.Path, None],
                                             affine: bool=False, transonly: bool=False) -> None:
    if affine and transonly:
        raise ValueError('Cannot set both affine and transonly flag. Choose one')
    
    reg = RobustRegister()
    reg.inputs.source_file = source
    reg.inputs.target_file = target
    reg.inputs.out_reg_file = out_lta_file
    reg.inputs.half_source = out_half_src_file
    reg.inputs.half_source_xfm = out_half_src_lta_file
    reg.inputs.half_targ = out_half_tgt_file
    reg.inputs.half_targ_xfm = out_half_tgt_lta_file
    reg.inputs.est_int_scale = True
    reg.inputs.auto_sens = True
    if transonly:
        reg.inputs.trans_only = True
    if affine:
        reg.inputs.args = '--affine'
    reg.run()

""" Wrapper to mri_vol2vol """
def freesurfer_resample_volumes(source: typing.Union[str, pathlib.Path], target: typing.Union[str, pathlib.Path], lta_file: typing.Union[str, pathlib.Path], 
                                output_file: typing.Union[str, pathlib.Path], interp:str='nearest') -> None:
    applyreg = ApplyVolTransform()
    applyreg.inputs.source_file = source
    applyreg.inputs.lta_file = lta_file
    applyreg.inputs.target_file = target
    applyreg.inputs.interp = interp
    applyreg.inputs.transformed_file = output_file
    applyreg.run()