#!/usr/bin/env python
from __future__ import print_function
import argparse
import os, sys, glob
import pydicom as dicom
import datetime
from rhscripts.dcm import replace_container

__scriptname__ = 'replace_dicom_container'
__version__ = '0.0.1'

"""
VERSIONING
  0.0.1 # Created script
"""

"""

Date: 16/4-2021
Author: Claes Ladefoged ( claes.noehr.ladefoged@regionh.dk )

###

Swap the PixelDate of one dicom dataset into another container dicom dataset

-----------------------------------------------------------------

### USAGE: replace_dicom_container.py [-h] [--series_number SERIES_NUMBER] [--series_description SERIES_DESCRIPTION] in_folder container out_folder

positional arguments:
  in_folder             Folder with new dicom files.
  container             Folder with container dicom files.
  out_folder            Folder with resulting dcm files.

optional arguments:
  -h, --help            show this help message and exit
  --series_number SERIES_NUMBER
                        SeriesNumber, used to match the files in the folders
  --series_description SERIES_DESCRIPTION
                        Replace the name of the series

-----------------------------------------------------------------

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert DICOM to MINC')

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("in_folder", help="Folder with new dicom files.")
    parser.add_argument("container", help="Folder with container dicom files.")
    parser.add_argument("out_folder", help="Folder with resulting dcm files.")
    parser.add_argument("--series_number", help="SeriesNumber, used to match the files in the folders", type=int)
    parser.add_argument("--series_description", help="Replace the name of the series", type=str)
    args = parser.parse_args()
    
    replace_container(in_folder=args.in_folder, container=args.container, out_folder=args.out_folder, 
                      SeriesNumber=args.series_number, SeriesDescription=args.series_description)


