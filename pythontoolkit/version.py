version = '0.0.10'
git_revision = ''

"""

VERSIONING
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

"""

def __show_version__():
	print('RH-scripts version: %s' % version)
	#print('Git revision: %s' % git_revision)