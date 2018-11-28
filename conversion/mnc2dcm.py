#!/usr/bin/env python
import argparse
from rhscripts.conversion import mnc_to_dcm
from rhscripts.version import __show_version__

__scriptname__ = 'mnc2dcm'
__version__ = '0.0.1'

"""

VERSIONING
  0.0.1 # Created script

"""

"""Convert a minc file to dicom

    Parameters
    ----------
    minc_file : string
        Path to the minc file
    dicom_container : string
        Path to the directory containing the dicom container
    dicom_output : string
        Path to the output dicom folder
    modify : boolean, optional
        Create new SeriesInstanceUID and SOPInstanceUID
        Default on if description or id is set
    description : string, optional
        Sets the SeriesDescription tag in the dicom files
    id : int, optional
        Sets the SeriesNumber tag in the dicom files
    verbose : boolean, optional
        Set the verbosity
    version : boolean, optional
        Print the version of the script
"""


parser = argparse.ArgumentParser()
parser.add_argument("minc_file", help="Path to the minc file to be converted", nargs='?')
parser.add_argument("dicom_container", help="Path to the folder containing dicom container files", nargs='?')
parser.add_argument("dicom_output", help="Path to the output folder for converted dicom", nargs='?')
parser.add_argument("-m","--modify", help="Modify DICOM headers to match container except for Instance Numbers", action="store_true")
parser.add_argument('--description', help="New name of the DICOM file", nargs=1, type=str)
parser.add_argument('--id', help="New ID of the DICOM file", nargs=1, type=int)
parser.add_argument("-v","--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("--version", help="Print version", action="store_true")

args = parser.parse_args()

if args.version:
    print('%s version: %s' % (__scriptname__,__version__))
    __show_version__()
    exit(-1)

if not args.minc_file or not args.dicom_container or not args.dicom_output:
    parser.print_help()
    print('Too few arguments')
    exit(-1)


mnc_to_dcm(args.minc_file, args.dicom_container, args.dicom_output, args.verbose, args.modify, args.description, args.id)