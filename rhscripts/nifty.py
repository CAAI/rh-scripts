from nipype.interfaces.fsl import Threshold, IsotropicSmooth, RobustFOV, BET, FLIRT
from nipype.interfaces.fsl.maths import ApplyMask, BinaryMaths
from nipype.interfaces.niftyreg import RegResample, RegAladin, RegTransform


######################################################################################################
#################################   Registration utilities   ##########################################
######################################################################################################


def reg_resample(ref_file, flo_file, trans_file, out_file, interpol='NN', pad_val=None):
    """Resample nifty file to a reference template given a transformation matrix

    Args:
        ref_file (a pathlike object or str): The input reference/target image
        flo_file (a pathlike object or str): The input floating/source image
        trans_file (a pathlike object or str): The input transformation matrix file
        out_file (a pathlike object or str): The output filename of the transformed image
        interpol (‘NN’ or ‘LIN’ or ‘CUB’ or ‘SINC’): Type of interpolation. Defaults to 'NN'.
        pad_val (float, optional): Padding value to pad. Defaults to None.
    """    
    
    rsl = RegResample()
    rsl.inputs.ref_file = ref_file
    rsl.inputs.flo_file = flo_file
    rsl.inputs.trans_file = trans_file
    rsl.inputs.inter_val = interpol
    if pad_val:
        rsl.inputs.pad_val = 0.0
    rsl.inputs.out_file = out_file
    rsl.run()
    
    
def reg_aladin(ref_file, flo_file, res_file, aff_file):
    """Block Matching algorithm for symmetric global registration

    Args:
        ref_file (a pathlike object or str): The input reference/target image
        flo_file (a pathlike object or str): The input floating/source image
        res_file (a pathlike object or str): The affine transformed floating image
        aff_file (a pathlike object or str): The output affine matrix file
    """ 
    
    ral = RegAladin()
    ral.inputs.ref_file = ref_file
    ral.inputs.flo_file = flo_file
    ral.inputs.res_file = res_file
    ral.inputs.aff_file = aff_file
    ral.run()
    

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


def iso_resample(in_file, out_file, voxel_size):
    """Resamples an existing volume with the target voxel size

    Args:
        in_file (a pathlike object or str): Input file
        out_file (a pathlike object or str): Output file
        voxel_size (float): target voxel sixe in mm
    """      
    resampler = FLIRT()
    resampler.inputs.in_file = in_file
    resampler.inputs.reference = in_file
    resampler.inputs.out_file = out_file
    resampler.inputs.output_type = "NIFTI_GZ"
    resampler.inputs.apply_isoxfm = voxel_size
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
    
def skull_strip(in_file, out_file, frac, mask=True):
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
    
def rescale(in_file, operand_value):
    """Multiplies the full volume with a numeric value

    Args:
        in_file (a pathlike object or str): Image to operate on
        operand_value (float): value to multiply the data
    """
    
    multiplier = BinaryMaths()
    multiplier.inputs.in_file = in_file
    multiplier.inputs.operation = 'mul'
    multiplier.inputs.operand_value = operand_value
    multiplier.inputs.out_file = in_file
    multiplier.run()
    
