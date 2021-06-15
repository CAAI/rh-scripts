#!/usr/bin/env python
import argparse
from rhscripts.conversion import rtx_to_nii
from rhscripts.version import __show_version__


__scriptname__ = 'rtx2nii'
__version__ = '0.0.1'

##
# RTX2NII python script
# VERSIONS:
#  - 0.0.1 :: 2021-06-15 : Added script as a copy of rtx2mnc

parser = argparse.ArgumentParser(description='RTX2MNC.')
parser.add_argument('RTX', help='Path to the DICOM RTX file', nargs='?')
parser.add_argument('container', help='Path to the MINC container file', nargs='?')
parser.add_argument('output', help='Path to the OUTPUT MINC RT file', nargs='?')
parser.add_argument('--behavior', help='Choose how to convert to polygon. Options: default, mirada',type=str,default='default')
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
#parser.add_argument("--copy_name", help="Copy the name of the RTstruct (defined in Mirada) to the tag dicom_0x0008:el_0x103e of the MNC file", action="store_true")
parser.add_argument("--version", help="Print version", action="store_true")

args = parser.parse_args()

if args.version:
	print('%s version: %s' % (__scriptname__,__version__))
	__show_version__()
	exit(-1)

if not args.RTX or not args.MINC or not args.RTMINC:
	parser.print_help()
	print('Too few arguments')
	exit(-1)

rtx_to_nii(dcmfile=args.RTX,
           nii_container_file = args.container,
           nii_output_file = args.output,
           behavior=args.behavior,
           verbose=args.verbose,
           copy_name=False) # not yet implemented
