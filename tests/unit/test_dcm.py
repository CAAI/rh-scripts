# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:23:44 2021

@author: clad0003
"""

import unittest
import pydicom
from rhscripts.dcm import Anonymize

class TestAnonymize(unittest.TestCase):
    
    def setUp(self):
        # Load test data
        self.ds = pydicom.read_file(pydicom.data.get_testdata_file("CT_small.dcm"))
        self.ds_untouched = pydicom.read_file(pydicom.data.get_testdata_file("CT_small.dcm"))
        
    def test_anonymize_fields(self):
        """
        Test that all fields are replaced or deleted after anonymization
        """
        
        # Set up class
        anon = Anonymize()
        
        # Anonymize
        ds = anon.anonymize_dataset( self.ds ) 
        
        # Check tags should be empty
        tags = ['PatientBirthDate','PatientAddress','PatientTelephoneNumbers']
        for tag in tags:
            if tag in self.ds_untouched:
                self.assertEqual( ds[tag].value, '' )
        
        # Check tags are replaced
        tags = ['PatientID','AccessionNumber','StudyID']
        for tag in tags:
            if tag in self.ds_untouched:
                self.assertNotEqual( ds[tag].value, self.ds_untouched[tag].value )
        
        # Check tags are removed
        tags_deleted = ['OtherPatientIDs', 'OtherPatientIDsSequence']
        for tag in tags_deleted:
            if tag in self.ds_untouched:
                self.assertTrue( tag not in ds )
                
                
    def test_uid_1( self ):
        """
        All UIDs should be replaced
        """
        
        # Set up class
        anon = Anonymize()
        
        # Anonymize
        ds = anon.anonymize_dataset( self.ds, replaceUIDs=True ) 
        
        self.assertNotEqual( ds.StudyInstanceUID, self.ds_untouched.StudyInstanceUID )
        self.assertNotEqual( ds.SeriesInstanceUID, self.ds_untouched.SeriesInstanceUID )
        self.assertNotEqual( ds.SOPInstanceUID, self.ds_untouched.SOPInstanceUID )
        
    def test_uid_2( self ):
        """
        Only SOP UID should be replaced
        """
        
        # Set up class
        anon = Anonymize()
        
        # Anonymize
        ds = anon.anonymize_dataset( self.ds, replaceUIDs=True, 
                                     studyInstanceUID=self.ds.StudyInstanceUID,
                                     seriesInstanceUID=self.ds.SeriesInstanceUID ) 
        
        self.assertEqual( ds.StudyInstanceUID, self.ds_untouched.StudyInstanceUID )
        self.assertEqual( ds.SeriesInstanceUID, self.ds_untouched.SeriesInstanceUID )
        self.assertNotEqual( ds.SOPInstanceUID, self.ds_untouched.SOPInstanceUID )
        
    def test_uid_3( self ):
        """
        Only Study UID should be kept
        """
        
        # Set up class
        anon = Anonymize()
        
        # Anonymize
        ds = anon.anonymize_dataset( self.ds, replaceUIDs=True, 
                                     studyInstanceUID=self.ds.StudyInstanceUID ) 
        
        self.assertEqual( ds.StudyInstanceUID, self.ds_untouched.StudyInstanceUID )
        self.assertNotEqual( ds.SeriesInstanceUID, self.ds_untouched.SeriesInstanceUID )
        self.assertNotEqual( ds.SOPInstanceUID, self.ds_untouched.SOPInstanceUID )
        
        
if __name__ == '__main__':
    unittest.main()