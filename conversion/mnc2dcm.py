#!/usr/bin/env python
import argparse
from rhscripts.conversion import mnc_to_dcm

parser = argparse.ArgumentParser()
parser.add_argument("minc_file", help="Path to the minc file to be converted")
parser.add_argument("dicom_container", help="Path to the folder containing dicom container files")
parser.add_argument("dicom_output", help="Path to the output folder for converted dicom")
parser.add_argument("-m","--modify", help="Modify DICOM headers to match container except for Instance Numbers", action="store_true")
parser.add_argument('--description', help="New name of the DICOM file", nargs=1, type=str)
parser.add_argument('--id', help="New ID of the DICOM file", nargs=1, type=int)
parser.add_argument("-v","--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

mnc_to_dcm(args.minc_file, args.dicom_container, args.dicom_output, args.verbose, args.modify, args.description, args.id)