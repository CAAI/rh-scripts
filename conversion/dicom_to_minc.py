#!/usr/bin/env python
import argparse
from rhscripts.conversion import dcm_to_mnc

parser = argparse.ArgumentParser(description='Convert DICOM to MINC')

parser.add_argument('dicom', type=str, help='Folder for dicom files')

parser.add_argument('--target', help='Destination dir',default='.')
parser.add_argument('--fname', help='Minc filename')
parser.add_argument('--dname', help='Folder for mincfiles')

args = parser.parse_args()

dcm_to_mnc(args.dicom,target=target,fname=args.fname,dname=args.dname)
