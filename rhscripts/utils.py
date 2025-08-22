#!/usr/bin/env python

import os
import itertools
import numpy as np
import sys, random, typing, time, pydicom
import pathlib
import typing
from pathlib import Path
import pandas as pd
import torchio as tio
from scipy.ndimage import binary_dilation
from skimage import measure
from rhscripts import nifty as RHnifty
import tempfile


def listdir_nohidden(path):
    """List dir without hidden files

    Parameters
    ----------
    path : string
        Path to folder with files
    """
    return [f for f in os.listdir(path) if not f.startswith('.')]

def bbox_ND(img):
    """Get bounding box for a mask with N dimensionality

    Parameters
    ----------
    img : numpy array or python matrix
    """
    N = img.ndim
    out = []
    for ax in itertools.combinations(range(N), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)

class LMParser:
    """ LMParser

    Inspired by: https://www.siemens-healthineers.com/en-us/molecular-imaging/petlink-documents

    # Logic is used by lmparser.py

    ### You can also load in this function to other script, e.g. this simple case:
        from rhscripts.utils import LMParser
        parser=LMParser('llm.ptd')
        parser.save_dicom('llm.dcm')
        parser.close()

    """
    def __init__( self, ptd_file: str, out_folder: str=None, anonymize: bool=False, verbose: bool=False ):
        self.start_time = time.time()
        # Constants
        self.LMDataID = "LARGE_PET_LM_RAWDATA"
        self.LMDataIDLen = len(self.LMDataID)
        self.LONG32BIT = 4
        self.BUFFERSIZE = 0x100 # 1 kb (c++ uses 1 mb, but this is much slower in python..)
        # Input args
        self.filename = Path( ptd_file )
        self.out_folder = Path(out_folder if out_folder is not None else self.filename.parent)
        self.out_folder.mkdir(parents=True,exist_ok=True)
        self.anonymize = anonymize
        self.verbose = verbose
        # Globals
        self.BytesReady = 0
        self.LMBuffer = None
        self.BytesToRead = None
        self.PROMPT, self.DELAY, self.KEEP, self.TOSS, self.EVENT_WORD, self.TAG_WORD = 0,0,0,0,0,0 # Counters
        self.do_chop=False
        # Setup and open PTD file
        self.__open_ptd_file()
        self.__read_dicom_header()
        self.__determine_bit_type()

    def __determine_bit_type( self ):
        ds = pydicom.filereader.dcmread(pydicom.filebase.DicomBytesIO(self.DicomBuffer))
        if (0x29, 0x1010) not in ds:
            return
        tag = ds[0x29, 0x1010]
        partial = tag.value
        for ind, l in enumerate(partial.decode().split('\n')):
            if l.startswith('%LM event and tag words format'):
                bit_type = str(l.split(':=')[1]).strip()
                self.is_32bit = bit_type == "32"
                self.__print(f"Found data with bittype: {bit_type}. So is_32bit is {self.is_32bit}")

    def __update_header( self ):
        import tempfile, re
        temp_filename = next(tempfile._get_candidate_names())

        old_dicombuffer_length = len(self.DicomBuffer)

        # Modify the line found with "cat/strings <.ptd>" called: "tracer activity at time of injection (Bq):=<dose>"
        ds = pydicom.filereader.dcmread(pydicom.filebase.DicomBytesIO(self.DicomBuffer))
        tag = ds[0x29, 0x1010]
        partial = tag.value
        start_string = 0
        end_string = 0
        for ind, l in enumerate(partial.decode().split('\n')):
            start_string = end_string
            end_string+=len(l)+1
            if l.startswith('tracer activity'):
                injected_dose_e = l.split(':=')[1].split('\r')[0]
                injected_dose = float(injected_dose_e)
                retained_injected_dose = injected_dose * float( self.retain / 100.0 )
                break
        self.__print('{:.1f} MBq -> {:.1f} MBq at {} retain value (in %)'.format(injected_dose/1000000, retained_injected_dose/1000000, self.retain))
        retained_injected_dose_e = '{:.3e}'.format(retained_injected_dose)
        new_string = l.replace(injected_dose_e, retained_injected_dose_e)+'\n'
        ds[0x29, 0x1010].value = partial.replace(partial[start_string:end_string], str.encode(new_string))
        ds.save_as(temp_filename)
        with open(temp_filename, "rb") as raw:
            self.DicomBuffer = raw.read(self.DicomHeaderLength)

        # Update the XML tag: <InjectedDose>...</InjectedDose>
        reg_str = "<InjectedDose>(.*?)</InjectedDose>"
        res = re.findall(reg_str, str(self.DicomBuffer))
        for dose in res:
            retained_injected_dose = int(float(dose) * float( self.retain / 100.0 ))
            retained_injected_dose = str(retained_injected_dose).zfill(len(dose.split('.')[0]))
            retained_injected_dose_trailing_zeros = "0" * len(dose.split('.')[1])
            dose_string = f"<InjectedDose>{dose}</InjectedDose>"
            retained_dose_string = f"<InjectedDose>{retained_injected_dose}.{retained_injected_dose_trailing_zeros}</InjectedDose>"
            if not len(retained_dose_string) == len(dose_string):
                # OBS - JSRecon12 fails if the the two numbers does not have the same number of digits, including before and after delimiter
                print("Retained dose does not have the same number of digits in XML tag. JSrecon will fail. Exiting")
                exit(-1)
            self.DicomBuffer = self.DicomBuffer.replace(str.encode(dose_string), str.encode(retained_dose_string))
        Path(temp_filename).unlink()

        self.DicomHeaderLength = len(self.DicomBuffer)
        self.__print(f"Modified DicomHeaderLength\n\tfrom {old_dicombuffer_length}\n\tto {self.DicomHeaderLength}")

    def chop( self, retain: int=None, out_filename: str=None, seed: int=11, random_scaling_method: str='default'):
        # Input args
        self.do_chop = True
        self.out_filename = out_filename
        self.retain = retain
        self.seed = seed
        # Open OutFile for writing
        self.OutFile = open( self.__generate_output_name() ,'wb')
        # Scale retain to 0-1.
        retain_fraction = float( self.retain / 100.0 )
        # Setup random
        random.seed(self.seed)
        # Setup LM file
        self.__prepare_lm_file()
        self.__print(f'Starting LMChopper with fraction={retain_fraction} and random_scaling_method={random_scaling_method}')

        if self.is_32bit:

            for word in self.__read_list():
                int_word = int.from_bytes(word,byteorder='little')
                if (int_word & 0x80000000) == 0x80000000:
                    self.TAG_WORD += 1
                    self.OutFile.write(word)
                    if (int_word >> 28 & 0xe) == 0x8:
                        if (listms := int_word & 0x1fffffff) > 0 and listms % 10000 == 0:
                            self.__print(f"Finished {listms/1000} seconds")
                else:
                    self.EVENT_WORD += 1

                    
                    if int_word >> 30 == 0x1:
                        self.PROMPT += 1
                    else:
                        self.DELAY += 1
                    
                    random_fraction = random.random()
                    if random_scaling_method.lower() == 'default':
                        # Allmost all tracers should go here
                        if random_fraction < retain_fraction:
                            self.OutFile.write(word)
                            self.KEEP += 1
                        else:
                            self.TOSS += 1
                    elif random_scaling_method.lower() == 'rb82':
                        # Scales randoms quadratically if tracer is Rb82
                        if (int_word >> 30 == 0x1 and random_fraction < retain_fraction)\
                        or ((int_word >> 30) != 0x1 and random_fraction < (retain_fraction)**2): 
                            self.OutFile.write(word)
                            self.KEEP += 1
                        else:
                            self.TOSS += 1
                    else:
                        raise NotImplementedError(random_scaling_method)
        else:
            raise NotImplemented('64bit LM chopping is not yet implemented.')
            # Below does not yet work.
            # TODO:
            # * Unsure if read_list() works for 64 bit?
            # * The while loops does not fit well with the python implementation..
            lmwordctr = 0
            wordok = 0

            word1 = next(self.__read_list())
            exit(-1)

            for word1 in self.__read_list():
                while not wordok:
                    Synch = 0
                    word0 = 0
                    while not Synch:
                        int_word1 = int.from_bytes(word1,byteorder='little')
                        if (int_word1 & 0x80000000 ) and ( word0 != 0 ):
                            Synch+=1
                        else:
                            word0 = word1.copy()
                    lmword0 = word0
                    lmword1 = word1
                    wordok += 1
                # if word is tag word, writethrough to all lists
                if( lmword0 & 0x40000000 ):
                    self.TAG_WORD += 1
                    self.OutFile.write(lmword0)
                    self.OutFile.write(lmword1)
                    lmwordctr += 1
                    if lmwordctr % 10000 == 0:
                        self.__print(f"Finished {lmwordctr} words")
                else:
                    self.EVENT_WORD += 1
                    if random.random() < retain_fraction:
                        self.OutFile.write(lmword0)
                        self.OutFile.write(lmword1)
                        self.KEEP += 1
                    else:
                        self.TOSS += 1



        self.__print("Done parsing LM words")
        self.__print(f"Prompts: {self.PROMPT}\nDelays: {self.DELAY}")
        self.__print(f"TAGS: {self.TAG_WORD}\nEVENTS: {self.EVENT_WORD}")
        self.__print(f"Keep: {self.KEEP}\nToss: {self.TOSS}\nRatio: {self.KEEP/self.EVENT_WORD*100:.2f}")

        # Modify DICOM header
        self.__update_header()

        # Write DICOM header etc back to file
        self.__write_header()

    def fake_chop( self, retain: int=None, out_filename: str=None ):
        self.out_filename = out_filename
        self.retain = retain
        self.do_chop = True # To trigger write of out file
        # Open OutFile for writing
        self.OutFile = open( self.__generate_output_name() ,'wb')
        # Scale retain to 0-1.
        retain_fraction = float( self.retain / 100.0 )

        # Write LM data
        self.__prepare_lm_file()
        for word in self.__read_list():
            self.OutFile.write(word)
        self.__print("Done parsing LM words")

        # Modify DICOM header
        self.__update_header()

        # Write DICOM header etc back to file
        self.__write_header()

    def return_LM_statistics( self ) -> pd.DataFrame:
        # Setup LM file
        self.__prepare_lm_file()

        dict_prompts = {0:0}
        dict_delays = {0:0}
        timestamp = 0
        for word in self.__read_list():
            int_word = int.from_bytes(word,byteorder='little')
            if (int_word & 0x80000000) == 0x80000000:
                # TAG WORD
                if (int_word >> 28 & 0xe) == 0x8:
                    if (listms := int_word & 0x1fffffff) > 0 and listms % 1000 == 0:
                        timestamp = listms/1000
                        dict_prompts[timestamp] = 0
                        dict_delays[timestamp] = 0
                        self.__print(f"Finished {listms/1000} seconds")
            else:
                # EVENT WORD

                if int_word >> 30 == 0x1:
                    dict_prompts[timestamp] += 1
                else:
                    dict_delays[timestamp] += 1
        df = pd.DataFrame(columns=['t','type','count'])
        for k,v in dict_prompts.items():
            df = df.append({'t': k, 'type':'prompt','numEvents':v},ignore_index=True)
            df = df.append({'t': k, 'type':'delay', 'numEvents':dict_delays[k]},ignore_index=True)
        df.t = df.t.astype('float')
        df.numEvents = df.numEvents.astype('int')
        self.__print("Done parsing LM words")
        return df

    def close( self ):
        self.LMFile.close()
        if self.do_chop:
            self.OutFile.close()
        self.__print("Closed files")
        self.__print("Done parsing in {:.0f} seconds".format( time.time()-self.start_time))

    def return_converted_dicom_header( self, anonymize_id: str=None, studyInstanceUID: str=None,
                                       seriesInstanceUID: str=None, replaceUIDs: bool=False ) -> pydicom.dataset.FileDataset:
        """
        Return the DICOM header from the PTD file.
        """
        ds = pydicom.dcmread( pydicom.filebase.DicomBytesIO( self.DicomBuffer ) )
        if self.anonymize:
            from rhscripts.dcm import Anonymize
            anon = Anonymize()
            ds = anon.anonymize_dataset(ds, new_person_name=anonymize_id,
                                        studyInstanceUID=studyInstanceUID,
                                        seriesInstanceUID=seriesInstanceUID,
                                        replaceUIDs=replaceUIDs)
        return ds

    def save_dicom( self, out_dicom: str ):
        dcm_file = self.out_folder.joinpath( out_dicom )
        self.return_converted_dicom_header().save_as( str( dcm_file.absolute() ) )
        self.__print("Saved DICOM to: {}".format(dcm_file))

    """ PRIVATE FUNCTIONS """

    def __set_relative_to_end( self, distance: int ) -> int:
        self.LMFile.seek( self.TotalLMFileSize - distance, os.SEEK_SET )

    def __read_dicom_header( self ):
        self.__set_relative_to_end( self.LMDataIDLen+self.LONG32BIT )
        self.DicomHeaderLength = int.from_bytes(self.LMFile.read(self.LONG32BIT),'little')
        self.__set_relative_to_end( self.DicomHeaderLength+self.LMDataIDLen+self.LONG32BIT )
        self.DicomBuffer = self.LMFile.read(self.DicomHeaderLength)
        self.__print(f"Read DICOM header of length {self.DicomHeaderLength}")

    def __open_ptd_file(self):
        # Open file
        if not self.filename.is_file(): sys.exit("No such PTD file",self.filename)
        self.LMFile = open(self.filename,'rb')
        self.__print(f"Opened LMFile {self.filename.name}")

        # Read total file size
        self.LMFile.seek(0,os.SEEK_END)
        self.TotalLMFileSize = self.LMFile.tell()

        # Check type
        self.__set_relative_to_end( self.LMDataIDLen )
        if not (PTDtype := self.LMFile.read(self.LMDataIDLen).decode('utf-8')) == self.LMDataID:
            sys.exit(f'String {PTDtype} does not match {self.LMDataID}')
        self.__print(f"PTD file is {PTDtype}")

    def __prepare_lm_file( self ):
        # Rewind and prepare for reading
        self.LMFile.seek(0)
        assert self.LMFile.tell() == 0

        # Get listmode-part size of file
        self.BytesRemaining = self.TotalLMFileSize - self.DicomHeaderLength - self.LMDataIDLen - self.LONG32BIT
        self.__print(f"Got LLM file size: {self.BytesRemaining}")

    def __write_header( self ):
        # Append DICOM header file
        self.OutFile.write( self.DicomBuffer ) # Could edit total dose in header here..
        # Write length of DICOM header
        self.OutFile.write( self.DicomHeaderLength.to_bytes(self.LONG32BIT, byteorder='little'))
        # Write LMDataID string
        self.OutFile.write( bytes(self.LMDataID,'ascii') )
        self.__print("Wrote header back to LLM file")

    def __generate_output_name( self ) -> str:
        return self.out_folder.joinpath(self.out_filename).absolute() if self.out_filename else \
               self.out_folder.joinpath('{}-{:.3f}.ptd'.format(self.filename.stem,self.retain)).absolute()

    def __read_list( self ) -> typing.Generator[bytes, None, None]:
        while self.BytesRemaining > 0 or self.BytesReady > 0:
            if self.BytesReady == 0:
                BytesToRead = self.BUFFERSIZE if self.BytesRemaining >= self.BUFFERSIZE else self.BytesRemaining
                self.LMBuffer = self.LMFile.read(BytesToRead)
                self.BytesReady = BytesToRead
                self.BytesRemaining -= BytesToRead
            word = self.LMBuffer[:self.LONG32BIT]
            self.LMBuffer = self.LMBuffer[self.LONG32BIT:]
            self.BytesReady -= self.LONG32BIT
            yield word

    def __read_file_backward(self):
        self.__set_relative_to_end(0)
        # Get the current position of pointer i.e eof
        pointer_location = self.LMFile.tell()
        # Create a buffer to keep the last read line
        buffer = bytearray()
        # Loop till pointer reaches the top of the file
        while pointer_location >= 0:
            # Move the file pointer to the location pointed by pointer_location
            self.LMFile.seek(pointer_location)
            # Shift pointer location by -1
            pointer_location = pointer_location -1
            # read that byte / character
            new_byte = self.LMFile.read(1)
            new_byte
            # If the read byte is not end of line character then continue reading
            if new_byte != b'\n':
                # If last read character is not eol then add it in buffer
                buffer.extend(new_byte)
            else:
                yield buffer
                # Reinitialize the byte array to save next line
                buffer = bytearray()
        # As file is read completely, if there is still data in buffer, then its the first line.
        if len(buffer) > 0:
            # Yield the first line too
            yield buffer
                
    def read_tail(self, stopword='DICM', return_full=False, strict=False, delimiter=':='):
        """
        Will parse the PTD file from the back (as when using strings ptd | tail -500).
        Will stop once the stopword has been found. Will try to parse xml format and KEY:=VALUE format.
        Use strict=True if the KEY:=VALUE must be parsed, otherwise, first instance will only be parsed
        """
        import re
        lines = []
        info = {}
        # read the LM file backward
        stopword = stopword[::-1]
        delimiter = delimiter[::-1]
        for byteline in self.__read_file_backward():
            line = ''
            for c in byteline:
                if 31 < c < 126: # or 190 < c < 255:
                    line += chr(c)
                elif c == 255:
                    line += ''
                elif 0 < c < 32:
                    if stopword in line:
                        break
                    if len(line) > 3:
                        lines.append(line[::-1].strip())
                    line = ''
                
            if len(line) > 3:
                lines.append(line[::-1].strip())
            if stopword in line:
                if not strict or delimiter in line:
                    break 

        # First parse lines, expecting XML format
        for l in lines:
            try:
                # parse info with pattern <info_name>info</info_name>
                x = re.findall("<(.*)>(.*)</", l)[0]
                info[x[0].strip()] = x[1].strip()
            except Exception as e:
                pass
        
        # Now parse expecting KEY:=VALUE format
        for l in lines:
            try:
                x = l.split(':=')
                info[x[0].strip()] = x[1].strip()
            except Exception as e:
                pass

        if stopword[::-1] in info and not return_full:
            return info[stopword[::-1]]
        else:
            return info

    def __print( self, message : str):
        if self.verbose:
            print( message )

def cluster_analysis_serial_data(baseline: typing.Union[str, pathlib.Path, tio.IMAGE], followup: typing.Union[str, pathlib.Path, tio.IMAGE],
                                 baseline_target: typing.Union[str, pathlib.Path, None], baseline_lta_file: typing.Union[str, pathlib.Path, None],
                                 followup_target: typing.Union[str, pathlib.Path, None], followup_lta_file: typing.Union[str, pathlib.Path, None],
                                 minimum_allowed_size_in_mm3: float=None, binary_dilation_steps_pr_cluster: int=1, 
                                 enlarged_percentage_threshold: float=0, return_longitudinal_file: bool=True,
                                 save_path_baseline_cc=None, save_path_followup_cc=None,
                                 save_path_baseline_cluster_halfspace=None, save_path_followup_cluster_halfspace=None):

    # Read files if they are not
    if type(baseline) in (str, pathlib.Path):
        baseline = tio.LabelMap(baseline)
    if type(followup) in (str, pathlib.Path):
        followup = tio.LabelMap(followup)

    if baseline_target is not None or followup_target is not None:
        temp_dir_object = tempfile.TemporaryDirectory()
        temp_dir = Path(temp_dir_object.name)
    
    spacings = []
    cluster_masks = []
    cluster_masks_orig = []
    volumes = []
    for ind, (tio_mask, target, lta_file) in enumerate(zip([baseline, followup], [baseline_target, followup_target], [baseline_lta_file, followup_lta_file])):
        mask_arr = tio_mask.numpy()[0]
        volume = {}
        mask_spacing = tio_mask.spacing
        spacings.append(mask_spacing)
        mask_cc = measure.label(mask_arr)
        
        unique, counts = np.unique(mask_cc, return_counts=True)

        for les in unique:
            if les == 0:
                continue
            les_size = counts[np.where((unique==les))] * mask_spacing[0]* mask_spacing[1]*mask_spacing[2] 
            if les_size < minimum_allowed_size_in_mm3: 
                mask_arr[np.where((mask_cc == les))] = 0
                mask_cc[np.where((mask_cc==les))] = 0
            else:
                volume[les] = les_size[0]
        volumes.append(volume)
        cluster_masks_orig.append(mask_cc)
        mask_arr[np.where((mask_cc == 0))] = 0 # Needed?

        # Save mask
        if ind==0:
            # Baseline
            binary_in_ptspace = temp_dir / 'source_bin_ptspace.nii.gz' if save_path_baseline_cc is None else save_path_baseline_cc
        else:
            binary_in_ptspace = temp_dir / 'target_bin_ptspace.nii.gz' if save_path_followup_cc is None else save_path_followup_cc
        tio.LabelMap(tensor=np.expand_dims(mask_arr,0), affine=tio_mask.affine).save(binary_in_ptspace)
     
        # Save mask and cc in halfspace if target and lta_file is supplied
        if target is not None and lta_file is not None:
            # Resample to halfspace
            binary_in_halfspace = temp_dir / 'source_bin_halfspace.nii.gz' if ind == 0 else temp_dir / 'target_bin_halfspace.nii.gz'
            RHnifty.freesurfer_resample_volumes(source=binary_in_ptspace, target=target, lta_file=lta_file, output_file=binary_in_halfspace, interp='nearest')

            # Save cluster
            cluster_in_ptspace = temp_dir / 'source_cluster_ptspace.nii.gz' if ind == 0 else temp_dir / 'target_cluster_ptspace.nii.gz'
            tio.LabelMap(tensor=np.expand_dims(mask_cc,0).astype(mask_arr.dtype), affine=tio_mask.affine).save(cluster_in_ptspace)
            if ind == 0:
                # Baseline
                cluster_in_halfspace = temp_dir / 'source_cluster_halfspace.nii.gz' if save_path_baseline_cluster_halfspace is None else save_path_baseline_cluster_halfspace
            else:
                # Followup
                cluster_in_halfspace = temp_dir / 'target_cluster_halfspace.nii.gz' if save_path_followup_cluster_halfspace is None else save_path_followup_cluster_halfspace
            print(mask_cc.dtype, mask_arr.dtype)
            RHnifty.freesurfer_resample_volumes(source=cluster_in_ptspace, target=target, lta_file=lta_file, output_file=cluster_in_halfspace, interp='nearest')

            cluster_masks.append(tio.LabelMap(cluster_in_halfspace).numpy()[0])
        else:
            cluster_masks.append(mask_cc)
        
    cluster1 = cluster_masks[0]
    cluster2 = cluster_masks[1]

    # Mapping clusters to each other
    remap_dict = [{},{}]
    remap_dict_vol = [{},{}]
    for cluster_id, (cluster, other_cluster) in enumerate(zip([cluster1, cluster2],[cluster2,cluster1])):
        for label_id in np.unique(cluster):
            if label_id == 0:
                continue

            cluster_single_label = cluster==label_id 
            cluster_single_label_dilated = binary_dilation(cluster_single_label.astype('int8'), iterations=binary_dilation_steps_pr_cluster)

            if (overlap_sum := np.sum(other_cluster[cluster_single_label_dilated]>0)) > 0:
                # We have overlap
                remap_dict[cluster_id][label_id] = [u for u in np.unique(other_cluster[cluster_single_label_dilated]) if u > 0]
                remap_dict_vol[cluster_id][label_id] = overlap_sum

    # Building longi file
    mask_longi = np.zeros_like(cluster_masks_orig[1])
    processed_label_ids = []
    number_of_clusters = 0
    number_of_new_lesions=0
    number_of_enlarged_lesions=0
    df_lesions = pd.DataFrame(columns=['label_id','volume_baseline','volume_followup','percentage_difference'])
    for label_id, label_vol in volumes[1].items():
        if label_id in remap_dict[1].keys():
            
            vol_0 = np.sum([volumes[0][_label_id] for _label_id in remap_dict[1][label_id]])

            # Add volume of other labels that are relabeled to this one
            grouped_label_ids = []
            for _label_id_in_0 in remap_dict[1][label_id]:
                grouped_label_ids = grouped_label_ids + remap_dict[0][_label_id_in_0]
            grouped_label_ids = list(np.unique(grouped_label_ids))

            if label_id in processed_label_ids:
                continue
            print(label_id,"existing")
            number_of_clusters+=1
            processed_label_ids = processed_label_ids + grouped_label_ids
            
            vol_1 = np.sum([volumes[1][_label_id] for _label_id in grouped_label_ids]) 
            perc=vol_1/vol_0

            df_lesions = pd.concat([df_lesions, pd.DataFrame({
                'label_id': label_id,
                'volume_baseline': vol_0,
                'volume_followup': vol_1,
                'percentage_difference': np.round((perc-1)*100,1)
            }, index=[0])], ignore_index=True)

            
            for _label_id in grouped_label_ids:
                if (perc)>(1+enlarged_percentage_threshold):
                    # Enlarged by 70%
                    if _label_id == label_id:
                        #print("\tEnlarged from",vol_0,"to",vol_1,f"({(perc-1)*100})")
                        pass
                    mask_longi[cluster_masks_orig[1]==_label_id] = 7
                    number_of_enlarged_lesions+=1                    
                else:
                    mask_longi[cluster_masks_orig[1]==_label_id] = 5
        else:
            # print(label_id,"new with volume:", label_vol)
            mask_longi[cluster_masks_orig[1]==label_id] = 6
            number_of_clusters+=1
            number_of_new_lesions+=1
            df_lesions = pd.concat([df_lesions, pd.DataFrame({
                'label_id': label_id,
                'volume_followup': volumes[1][label_id],
            }, index=[0])], ignore_index=True)

    # Add volumes that are missing in followup
    for label_id, label_vol in volumes[0].items():
        if not label_id in remap_dict[0].keys():
            df_lesions = pd.concat([df_lesions, pd.DataFrame({
                'volume_baseline': label_vol
            }, index=[0])], ignore_index=True)

    # Clean up
    temp_dir_object.cleanup()

    if return_longitudinal_file:
        longitudinal_file = tio.LabelMap(tensor=np.expand_dims(mask_longi,0).astype('uint8'), affine=followup.affine)
        return df_lesions, longitudinal_file
    else:
        return df_lesions

    
