#!/usr/bin/env python3

from rhscripts.utils import LMParser
import argparse

__scriptname__ = 'lmparser'
__version__ = '0.0.1'

"""

VERSIONING
  0.0.1 # Added functionality similar to LMChopper and LMBreakout merged together.

"""


"""Convert a minc file to dicom

Parameters
----------
...
version : boolean, optional
    Print the version of the script
"""

""" LMParser 
Author: Claes Ladefoged
Date: 11-04-2021

### Example use cases from cmd-line ###
    
    Check LLM file is LISTMODE:
        python lmparser.py <ptd file>
    
    Extract DICOM header to file:
        python lmparser.py <ptd file> --out_dicom <dicom_filename>
        
    Chop LM file
        python lmparser.py <ptd file> <percent retained>
        
    Above will output dicom and chopped ptd file in same folder as input ptd.    
    You can specify output folder and/or output chopped name as optional inputs.
    
    Full example with arguments
        python lmparser.py llm.ptd --verbose --retain 25 --out_folder . \
               --out_filename lm_25p.ptd --out_dicom llm.dcm -seed 1337
---------------------------------------------------------------------------------------

"""

# INPUTS
parser = argparse.ArgumentParser()
parser.add_argument("ptd_file", help='Input PTD LLM file', type=str)
parser.add_argument("--retain", help='Percent (float) of LMM events to retain (0-100)', type=float)
parser.add_argument("--fake_retain", help='Percent (float) of LMM events to retain (0-100). !! Does not actually do any chopping !!, but update header of ptd to reflect the previously performed chop.', type=float)
parser.add_argument("--out_folder", help='Output folder for chopped PTD LLM file(s)', type=str)
parser.add_argument("--out_filename", help='Output filename for chopped PTD LLM file', type=str)
parser.add_argument("--seed", help='Seed value for random', default=11, type=int)
parser.add_argument("--out_dicom", help='Save DICOM header to file', type=str)
parser.add_argument('--anonymize', action='store_true')
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

parser = LMParser( ptd_file = args.ptd_file,  out_folder = args.out_folder, 
                   anonymize = args.anonymize, verbose = args.verbose)
if args.retain: parser.chop(retain = args.retain, out_filename = args.out_filename, seed = args.seed)
if args.fake_retain: parser.fake_chop(retain = args.fake_retain, out_filename = args.out_filename)
if args.out_dicom: parser.save_dicom(args.out_dicom)
parser.close()
    
