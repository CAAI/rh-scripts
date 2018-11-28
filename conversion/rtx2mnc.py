#!/usr/bin/env python

import numpy as np
try:
    import pydicom as dicom
except ImportError:
    import dicom
from dicom.filereader import InvalidDicomError
import matplotlib.pyplot as plt
import pyminc.volumes.volumes as pyvolume
import pyminc.volumes.factory as pyminc
import argparse
import cv2
from matplotlib.path import Path
import os


__scriptname__ = 'rtx2mnc'
__version__ = '0.0.1'

##
# RTX2MNC python script
# VERSIONS:
#  - 1.0.0 :: 2018-03-08 :: Added basic functionality working for one or more RT-files
#  - 1.0.1 :: 2018-03-08 :: Added RT-name in MNC header
#  - 1.0.2 :: 2018-04-10 :: BUG - slice location rounded incorrectly. Fixed.
##
# TODO:
#  - Add check for MINC-file matches RTX dimensions and IDs
##

parser = argparse.ArgumentParser(description='RTX2MNC.')
parser.add_argument('RTX', help='Path to the DICOM RTX file', nargs='?')
parser.add_argument('MINC', help='Path to the MINC container file', nargs='?')
parser.add_argument('RTMINC', help='Path to the OUTPUT MINC RT file', nargs='?')
parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("--visualize", help="Show plot of slices for debugging", action="store_true")
parser.add_argument("--copy_name", help="Copy the name of the RTstruct (defined in Mirada) to the name of the MNC file", action="store_true")
parser.add_argument("--version", help="Print version", action="store_true")

args = parser.parse_args()

if args.version:
	print('%s version: %s' % (__scriptname__,__version__))
	exit(-1)

if not dcmfile or not mnc_container_file or not mnc_output_file:
	parser.print_help()
	print('Too few arguments')
	exit(-1)

try:
	RTSS = dicom.read_file(dcmfile) 
	print(RTSS.StructureSetROISequence[0].ROIName)
	ROIs = RTSS.ROIContourSequence

	if args.verbose:
		print("Found",len(ROIs),"ROIs")

	volume = pyminc.volumeFromFile(mnc_container_file)

	for ROI_id,ROI in enumerate(ROIs):

		# Create one MNC output file per ROI
		RTMINC_outname = mnc_output_file if len(ROIs) == 1 else mnc_output_file[:-4] + "_" + str(ROI_id) + ".mnc"
		RTMINC = pyminc.volumeLikeFile(mnc_container_file,RTMINC_outname)
		contour_sequences = ROI.ContourSequence

		if args.verbose:
			print(" --> Found",len(contour_sequences),"contour sequences for ROI:",RTSS.StructureSetROISequence[ROI_id].ROIName)

		for contour in contour_sequences:
			assert contour.ContourGeometricType == "CLOSED_PLANAR"

			current_slice_i_print = 0
			
			if args.verbose:
				print("\t",contour.ContourNumber,"contains",contour.NumberOfContourPoints)

			world_coordinate_points = np.array(contour.ContourData)
			world_coordinate_points = world_coordinate_points.reshape((contour.NumberOfContourPoints,3))
			current_slice = np.zeros((volume.getSizes()[1],volume.getSizes()[2]))
			voxel_coordinates_inplane = np.zeros((len(world_coordinate_points),2))
			current_slice_i = 0
			for wi,world in enumerate(world_coordinate_points):
				voxel = volume.convertWorldToVoxel([-world[0],-world[1],world[2]])
				current_slice_i = voxel[0]
				voxel_coordinates_inplane[wi,:] = [voxel[2],voxel[1]]
			current_slice_inner = np.zeros((volume.getSizes()[1],volume.getSizes()[2]),dtype=np.float)
			converted_voxel_coordinates_inplane = np.array(np.round(voxel_coordinates_inplane),np.int32)
			cv2.fillPoly(current_slice_inner,pts=[converted_voxel_coordinates_inplane],color=1)
			p = Path(voxel_coordinates_inplane)
			points = np.array(np.nonzero(current_slice_inner)).T
			grid = p.contains_points(points[:,[1,0]])
			for pi,point in enumerate(points):
				if not grid[pi]:
					# REMOVE EDGE POINT BECAUSE CENTER IS NOT INCLUDED
					current_slice_inner[point[0],point[1]] = 0 

					if args.visualize:
						plt.plot(point[1],point[0],'bx')

				elif args.visualize:
					plt.plot(point[1],point[0],'bo')

			if args.visualize:
				plt.imshow(current_slice_inner)
				plt.plot(voxel_coordinates_inplane[:,0],voxel_coordinates_inplane[:,1],'ro')
				plt.show()

			RTMINC.data[int(round(current_slice_i))] += current_slice_inner 


		# Remove even areas - implies a hole.
		RTMINC.data[RTMINC.data % 2 == 0] = 0

		RTMINC.writeFile()
		RTMINC.closeVolume()

		if args.copy_name:
			print('minc_modify_header -sinsert dicom_0x0008:el_0x103e="'+RTSS.StructureSetROISequence[ROI_id].ROIName+'" '+RTMINC_outname)
			os.system('minc_modify_header -sinsert dicom_0x0008:el_0x103e="'+RTSS.StructureSetROISequence[ROI_id].ROIName+'" '+RTMINC_outname)

	volume.closeVolume()

except InvalidDicomError:
	print("Could not read DICOM RTX file",args.RTX)
	exit(-1)
