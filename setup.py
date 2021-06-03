#!/usr/bin/env python3

from setuptools import setup, find_packages
from rhscripts.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
     name='rhscripts',
     version=__version__,
     author="Claes Ladefoged",
     author_email="claes.noehr.ladefoged@regionh.dk",
     description="Scripts used at CAAI",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/CAAI/rh-scripts",
     scripts=[
             'dicom/anonymize_dicom.py',
             'dicom/replace_dicom_container.py',
             'conversion/dicom_to_minc.py',
  	    	 'conversion/rtx2mnc.py',
  	    	 'conversion/mnc2dcm.py',
             'conversion/nii2dcm.py',
  	    	 'conversion/rtdose2mnc.py',
  	    	 'conversion/hu2lac.py',
          	 'conversion/lac2hu.py',
             'utils/lmparser.py'
    ],
     packages=setuptools.find_packages(),
     install_requires=[
         'pyminc',
         'pydicom',
         'opencv-python',
         'matplotlib',
         'pandas',
         'nipype',
         'scikit-image'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
 )
