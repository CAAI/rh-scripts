#!/usr/bin/env python3

from rhscripts.metrics import dice_similarity
import argparse
import nibabel as nib

__scriptname__ = 'dice_similarity'
__version__ = '0.0.1'

"""

VERSIONING
  0.0.1 # Added functionality

"""


"""Calculate DICE score between two nifti files

Parameters
----------
...
version : boolean, optional
    Print the version of the script
"""

""" dice_similarity 
Author: Claes Ladefoged
Date: 31-10-2024

### Example use cases from cmd-line ###
    
    Get overall dice score, assuming binary input:
        python dice_similarity.py <nii_file_1> <nii_file_2>
    
    Apply a threshold on the images first:
        python dice_similarity.py <nii_file_1> <nii_file_2> --threshold <threshold>
        
    Use a mask to limit the threshold area:
        python dice_similarity.py <nii_file_1> <nii_file_2> --mask <nii_file_mask>
        
---------------------------------------------------------------------------------------

"""

# INPUTS
parser = argparse.ArgumentParser()
parser.add_argument("nii_file_1", help='Input nii file', type=str)
parser.add_argument("nii_file_2", help='Input nii file', type=str)
parser.add_argument("--threshold", help='Apply a threshold to the images', type=float)
parser.add_argument("--mask", help='Input mask file to limit the area', type=str)
args = parser.parse_args()

arr1 = nib.load(args.nii_file_1).get_fdata()
arr2 = nib.load(args.nii_file_2).get_fdata()

if args.threshold:
    arr1 = (arr1 > args.threshold).astype(int)
    arr2 = (arr2 > args.threshold).astype(int)
    
if args.mask:
    mask = nib.load(args.mask).get_fdata()
    arr1[mask<1] = 0
    arr2[mask<1] = 0 

dsc = dice_similarity(arr1, arr2)
print(dsc)
    
