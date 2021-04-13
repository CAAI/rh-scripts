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

if __name__ == '__main__':
    unittest.main()