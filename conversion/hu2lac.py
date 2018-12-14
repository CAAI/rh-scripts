#!/usr/bin/env python
import argparse
from rhscripts.conversion import hu2lac
from rhscripts.version import __show_version__


__scriptname__ = 'hu2lac'
__version__ = '0.0.1'

parser = argparse.ArgumentParser(description='Convert CT-HU to LAC')
parser.add_argument('infile', help='Path to the minc input file', nargs='?')
parser.add_argument('outfile', help='Path to the minc output file', nargs='?')
parser.add_argument("kvp", help="Integer that specify the kVp on CT scan", type=int, nargs='?')
parser.add_argument("--mrac", help="if set, scales the LAC [cm^-1] by 10000", action="store_true")
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("--version", help="Print version", action="store_true")

args = parser.parse_args()

if args.version:
	print('%s version: %s' % (__scriptname__,__version__))
	__show_version__()
	exit(-1)

if not args.infile or not args.outfile or not args.kvp:
	parser.print_help()
	print('Too few arguments')
	exit(-1)

hu2lac(args.infile, args.outfile,args.kvp,mrac=args.mrac,verbose=args.verbose)