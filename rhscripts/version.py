__version__ = '0.1.10'
git_revision = ''

"""

VERSIONING (UPDATED WHEN ADDING TO MASTER BRANCH)
  0.0.1 # Added conversion script and minimal example functions
  0.0.2 # Added mnc2dcm functionality
  0.0.3 # Added comments to pythontoolkit
  0.0.4 # Added utils and updated mnc2dcm
  0.0.5 # Added combability for UBUNTU pydicom in conversion
  0.0.6 # Added dicom as toolkit option
  0.0.7 # Added rtdose_to_mnc and rtx_to_mnc
  0.0.8 # Fixed bug with dicom toolkit conflicting with pydicom when installed with version < 1.0
  0.0.9 # Restructured mnc2dcm and moved LargestPixelValue into modify part of script
  0.0.10 # Added dry_run, roi_name, and crop_area functionality to conversion.rtx2mnc
  0.0.11 # mnc2dcm fixed with PET values above LargestPixelValue 32767
  0.0.12 # mnc2dcm4D fixed with PET values above LargestPixelValue 32767
  0.1.0 # Added hu2lac and lac2hu
  0.1.1 # Merging develop_claes with mnc2dcm functionality updates handle PET images
  0.1.2 # Added LMParser that can read and parse LLM files
  0.1.3 # Added DICOM anonymize functionality
  0.1.4 # Added anonymize function to LMParser
  0.1.5 # LMParser can return number of prompt/delayed events over time
  0.1.6 # Added replace_dicom_container function
  0.1.7 # Added nifty_to_dcm. Function merged with mnc_to_dcm* into to_dcm
  0.1.8 # Fixed RescaleSlope and RescaleIntercept, incl moving them in pr slice rather than global
  0.1.9 # Added to_rtx and read_rtx
  0.1.10 # Added hd_bet to nifti, fixed bugs in plotting and dcm.
         # Removed CMAKE as install option.
"""

def __show_version__():
	print('RH-scripts version: %s' % __version__)
	#print('Git revision: %s' % git_revision)
