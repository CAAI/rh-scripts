#!/usr/bin/env python

import os
try:
    import pydicom as dicom
except ImportError:
    import dicom

def look_for_dcm_files(folder):
	if findExtension(folder) != -1:
		return folder
	for root,subdirs,files in os.walk(folder):
		if len(subdirs) > 0:
			continue
		if not len(files) > 0:
			continue
		if findExtension(root) != -1:
			return root
	return -1
		
def dcm_to_mnc(folder,target='.',fname=None,dname=None,verbose=False,checkForFileEndings=True):
	dcmcontainer = look_for_dcm_files(folder) if checkForFileEndings else folder
	
	if dcmcontainer == -1:
		print("Could not find dicom files in container..")
		exit(-1)

	cmd = 'dcm2mnc -usecoordinates -clobber '+dcmcontainer+'/* '+target
	if not fname is None:
		cmd += ' -fname "'+fname+'"'
	if not dname is None:
		cmd += ' -dname '+dname

	if verbose:
		print("Command %s" % cmd)

	os.system(cmd)