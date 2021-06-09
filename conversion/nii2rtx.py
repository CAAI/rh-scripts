#!/usr/bin/env python
import argparse
from rhscripts.conversion import nii_to_rtx
from rhscripts.version import __show_version__

__scriptname__ = 'mnc2rtx'
__version__ = '0.0.1'

"""

VERSIONING
  0.0.1 # Created script

"""

"""Convert a nifty file to dicom

    Parameters
    ----------
    nifty_file : string
        Path to the nifty file. Assume integer values of [0,1(,..)]
    dicom_container : string
        Path to dicom container to be used
    dicom_output : string
        Name of the output folder
    out_filename : string
        Name of the output dicom file
    verbose : boolean, optional
        Verbosity of function
    verbose : boolean, optional
        Set the verbosity
    version : boolean, optional
        Print the version of the script
"""


parser = argparse.ArgumentParser()
parser.add_argument("nifty_file", help="Path to the nifty file to be converted", nargs='?')
parser.add_argument("dicom_container", help="Path to the folder containing dicom container files", nargs='?')
parser.add_argument("dicom_output", help="Path to the output folder for converted dicom", nargs='?')
parser.add_argument("out_filename", help="Name of the dicom file (without .dcm)", nargs='?')
parser.add_argument("-v","--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("--version", help="Print version", action="store_true")

args = parser.parse_args()

if args.version:
    print('%s version: %s' % (__scriptname__,__version__))
    __show_version__()
    exit(-1)

if not args.nifty_file or not args.dicom_container or not args.dicom_output or not args.out_filename:
    parser.print_help()
    print('Too few arguments')
    exit(-1)

nii_to_rtx( niifile=args.nifty_file,
            dcmcontainer=args.dicom_container,
            out_folder=args.dicom_output,
            out_filename=args.out_filename,
            verbose=args.verbose):
