#!/usr/bin/env python3
import argparse
from rhscripts.conversion import dosedcm_to_mnc
from rhscripts.version import __show_version__

__scriptname__ = 'rtdose2mnc'
__version__ = '0.0.1'

parser = argparse.ArgumentParser(description='Convert RD DICOM (dose distribution) to MINC')
parser.add_argument('file_dcm', type=str, help='The input RD DICOM file', nargs='?')
parser.add_argument('file_mnc', type=str, help='The output MINC file', nargs='?')
parser.add_argument("--version", help="Print version", action="store_true")
 
args = parser.parse_args()

if args.version:
    print('%s version: %s' % (__scriptname__,__version__))
    __show_version__()
    exit(-1)

if not args.file_dcm or not args.file_mnc:
    parser.print_help()
    print('Too few arguments')
    exit(-1)

dosedcm_to_mnc(args.file_dcm,args.file_mnc)