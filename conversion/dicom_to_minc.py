#!/usr/bin/env python
import argparse
from rhscripts.conversion import dcm_to_mnc
from rhscripts.version import __show_version__

__scriptname__ = 'dicom_to_minc'
__version__ = '0.0.1'

parser = argparse.ArgumentParser(description='Convert DICOM to MINC')

parser.add_argument('dicom', type=str, help='Folder for dicom files', nargs='?')

parser.add_argument('--target', help='Destination dir',default='.')
parser.add_argument('--fname', help='Minc filename')
parser.add_argument('--dname', help='Folder for mincfiles')
parser.add_argument("--version", help="Print version", action="store_true")

args = parser.parse_args()

if args.version:
	print('%s version: %s' % (__scriptname__,__version__))
	__show_version__()
	exit(-1)

if not args.dicom:
	parser.print_help()
	print('Too few arguments')
	exit(-1)

dcm_to_mnc(args.dicom,target=target,fname=args.fname,dname=args.dname)



