#!/usr/bin/env python
import argparse
from rhscripts.conversion import rtx_to_mnc
from rhscripts.version import __show_version__


__scriptname__ = 'rtx2mnc'
__version__ = '0.0.1'

##
# RTX2MNC python script
# VERSIONS:
#  - 1.0.0 :: 2018-03-08 :: Added basic functionality working for one or more RT-files
#  - 1.0.1 :: 2018-03-08 :: Added RT-name in MNC header
#  - 1.0.2 :: 2018-04-10 :: BUG - slice location rounded incorrectly. Fixed.
##
# TODO:
#  - Add check for MINC-file matches RTX dimensions and IDs
##

parser = argparse.ArgumentParser(description='RTX2MNC.')
parser.add_argument('RTX', help='Path to the DICOM RTX file', nargs='?')
parser.add_argument('MINC', help='Path to the MINC container file', nargs='?')
parser.add_argument('RTMINC', help='Path to the OUTPUT MINC RT file', nargs='?')
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
#parser.add_argument("--visualize", help="Show plot of slices for debugging", action="store_true")
parser.add_argument("--copy_name", help="Copy the name of the RTstruct (defined in Mirada) to the name of the MNC file", action="store_true")
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

rtx_to_mnc(args.RTX, args.MINC, args.RTMINC, args.verbose, args.copy_name)