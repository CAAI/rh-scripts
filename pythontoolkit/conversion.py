#!/usr/bin/env python

import os
try:
    import pydicom as dicom
except ImportError:
    import dicom


def findExtension(sourcedir,extensions = [".ima", ".IMA", ".dcm", ".DCM"]):
    """Return the number of files with one of the extensions, 
    or -1 no files were found, or if more than one type of extension is found

    Parameters
    ----------
    sourcedir : string
        Path to the directory to crawl for files with extensions
    extensions : string list, optional
        Extensions to look for, each mutually exclusive

    Notes
    -----
    If none of the folders in sourcedir contains the extensions

    Examples
    --------
    >>> import numpy.matlib
    """
    counts = [0,0,0,0]
    c = 0
    for ext in extensions:
        files = glob.glob(os.path.join(sourcedir,'*' + ext) )
        counts[c] = len(files)
        c += 1
    if sum(counts) > max(counts) or sum(counts) == 0:
        return -1
    else:
        return extensions[counts.index(max(counts))]

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

def mnc_to_dcm(mncfile,dicomcontainer,dicomfolder,verbose=False,modify=False,description=None,id=None):  
    ## TODO
    # Add slope and intercept
    # Fix max in numpy conversion
    # 4D MRI
    
    if description or id:
        modify = True
    
    dcmcontainer = look_for_dcm_files(dicomcontainer)
    if dcmcontainer == -1:
        print("Could not find dicom files in container..")
        exit(-1)

    #print dcmcontainer,listdir_nohidden(dcmcontainer)
    firstfile = listdir_nohidden(dcmcontainer)[0]
    try:
        ds=dicom.read_file(os.path.join(dcmcontainer,firstfile).decode('utf8'))
    except AttributeError:
        ds=dicom.read_file(os.path.join(dcmcontainer,firstfile))
    minc = volumeFromFile(mncfile)
    SmallestImagePixelValue = minc.data.min()
    LargestImagePixelValue = minc.data.max()
    np_minc = np.array(minc.data,dtype=ds.pixel_array.dtype)
    minc.closeVolume()

    ## Prepare for MODIFY HEADER
    try:
        newSIUID = unicode(datetime.datetime.now())
    except:
        newSIUID = str(datetime.datetime.now())
    newSIUID = newSIUID.replace("-","")
    newSIUID = newSIUID.replace(" ","")
    newSIUID = newSIUID.replace(":","")
    newSIUID = newSIUID.replace(".","")
    newSIUID = '1.3.12.2.1107.5.2.38.51014.' + str(newSIUID) + '11111.0.0.0' 

    negative_handled = False
    if( np.issubdtype(np.uint16, ds.pixel_array.dtype) and SmallestImagePixelValue < 0):
        if verbose:
            print("Ran into negative values in uint16 dtype, clamping to 0")
        np_minc = np.maximum( np_minc, 0 )
        negative_handled = True

    if verbose and SmallestImagePixelValue < 0 and not negative_handled:
        print("Unhandled dtype for negative values: %s" % ds.pixel_array.dtype)

    if np.max(np_minc) > LargestImagePixelValue:
        if verbose:
            print("Maximum value exceeds LargestImagePixelValue - setting to zero")
        np_minc[ np.where( np_minc > LargestImagePixelValue ) ] = 0

    if not os.path.exists(dicomfolder):
        os.mkdir(dicomfolder)

    #print len(listdir_nohidden(dcmcontainer)) , np_minc.shape[0]
    assert len(listdir_nohidden(dcmcontainer)) == np_minc.shape[0]

    if verbose:
            print("Converting to DICOM")
    if modify:
        if verbose:
            print("Modifying DICOM headers")

    for f in listdir_nohidden(dcmcontainer):
        try:
            ds=dicom.read_file(os.path.join(dcmcontainer,f).decode('utf8'))
        except AttributeError:
            ds=dicom.read_file(os.path.join(dcmcontainer,f))
        i = int(ds.InstanceNumber)-1

        assert ds.pixel_array.shape == (np_minc.shape[1],np_minc.shape[2])

        ds.LargestImagePixelValue = LargestImagePixelValue
        ds.PixelData = np_minc[i,:,:].tostring()

        if modify:
            ds.SeriesInstanceUID = newSIUID
            if not description == None:
                ds.SeriesDescription = description
            if not id == None:
                ds.SeriesNumber = id

            try:
                newSOP = unicode(datetime.datetime.now())
            except:
                newSOP = str(datetime.datetime.now())
            
            newSOP = newSOP.replace("-","")
            newSOP = newSOP.replace(" ","")
            newSOP = newSOP.replace(":","")
            newSOP = newSOP.replace(".","")
            newSOP = '1.3.12.2.1107.5.2.38.51014.' + str(newSOP) + str(i+1)
            ds.SOPInstanceUID = newSOP

        fname = "dicom_%04d.dcm" % int(ds.InstanceNumber)
        ds.save_as(os.path.join(dicomfolder,fname))

    if verbose:
        print("Output written to %s" % dicomfolder)


def dosedcm_to_mnc(dcm,mnc):
    """
    Created on Nov 28 2018
    Author: Anders Olin ( anders.olin@regionh.dk )

    ###

    Script to convert RD dose distributions to MINC
    -----------------------------------------------------------------
    usage: rtdose2mnc.py [-h] file.dcm file.mnc

    Convert DICOM to MINC

    positional arguments:
      file_dcm     Input (RD) DICOM file
      file_mnc     Output MINC file

    optional arguments:
      -h, --help   show this help message and exit
      --name NAME  Name instead of patient name

    -----------------------------------------------------------------

    """

    # Load the dicom
    ds = pydicom.dcmread(dcm)
    
    # Extract the starts and steps of the x,y,z space
    starts = ds.ImagePositionPatient
    steps = [float(i) for i in ds.PixelSpacing];
    if not (ds.SliceThickness==''):
        dz = ds.SliceThickness
    elif 'GridFrameOffsetVector' in ds: 
        dz = ds.GridFrameOffsetVector[1] -ds.GridFrameOffsetVector[0]
    else:
        raise IOError("Cannot determine slicethickness!")
    steps.append(dz)
    
    #reorder the starts and steps
    myorder = [2,1,0]
    starts = [ starts[i] for i in myorder]
    myorder = [2,0,1]
    steps = [ steps[i] for i in myorder]
    #change the sign (e.g. starts=[1,-1,-1].*starts)
    starts = [a*b for a,b in zip([1,-1,-1],starts)]
    steps = [a*b for a,b in zip([1,-1,-1],steps)]
    
    #Get the pixel data and scale it correctly
    dose_array = ds.pixel_array*float(ds.DoseGridScaling)
    
    # Write the output minc file
    out_vol = pyminc.volumeFromData(mnc,dose_array,dimnames=("zspace", "yspace", "xspace"),starts=starts,steps=steps)
    out_vol.writeFile() 
    out_vol.closeVolume() 