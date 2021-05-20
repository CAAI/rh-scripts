#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 12:47:55 2021

@author: claes
"""

import unittest
from rhscripts.utils import getLesionLevelDetectionMetrics
import numpy as np

# Helper functions
def precision(TP,FP):
    return TP/(TP+FP)
def recall(TP,FN):
    return TP/(TP+FN)
def f1(TP,FP,FN):
    return 2*TP/(2*TP+FP+FN)
    
class TestUtilMetrics(unittest.TestCase):
    
    def setUp(self):
        # Generate data
        self.reference = np.zeros((64,64,64))
        # Insert 2D region
        self.reference[10:20,10:20,32] = 1
        # Insert 3D region
        self.reference[40:50,40:50,40:50] = 1
        # Insert 3D edge region
        self.reference[:3,:4,:2] = 1
        
        # Perfect overlap
        self.predicted_perfect = self.reference.copy()
        
        # 2/3 lesions detected, 1 FN
        self.predicted_2of3 = np.zeros((64,64,64))
        self.predicted_2of3[10:20,10:20,32] = 1
        self.predicted_2of3[40:50,40:50,40:50] = 1
        
        # 2/3 lesions detected, plus 1 FP
        self.predicted_2of3TP_1FP = np.zeros((64,64,64))
        self.predicted_2of3TP_1FP[10:20,10:20,32] = 1
        self.predicted_2of3TP_1FP[40:50,40:50,40:50] = 1
        self.predicted_2of3TP_1FP[60:,60:,62] = 1     
        
    def test_only_TP(self):
        """
        All should be detected, no extra
        """
        
        metrics = getLesionLevelDetectionMetrics( reference_image=self.reference, predicted_image=self.predicted_perfect )
        
        # Expected values
        TP = 3
        FN = 0
        FP = 0
        self.assertEqual( metrics.precision, precision(TP=TP,FP=FP) )
        self.assertEqual( metrics.recall, recall(TP=TP,FN=FN) )
        self.assertEqual( metrics.f1, f1(TP=TP,FP=FP,FN=FN) )
        self.assertEqual( metrics.TP, TP )
        self.assertEqual( metrics.FP, FP )          
        
    def test_2TP_1FN(self):
        """
        2/3 lesions detected, 1 FN
        """
        
        metrics = getLesionLevelDetectionMetrics( reference_image=self.reference, predicted_image=self.predicted_2of3 )
        
        # Expected values
        TP = 2
        FN = 1
        FP = 0
        self.assertEqual( metrics.precision, precision(TP=TP,FP=FP) )
        self.assertEqual( metrics.recall, recall(TP=TP,FN=FN) )
        self.assertEqual( metrics.f1, f1(TP=TP,FP=FP,FN=FN) )
        self.assertEqual( metrics.TP, TP )
        self.assertEqual( metrics.FP, FP )    
        
    def test_2TP_1FP(self):
        """
        2/3 lesions detected, plus 1 FP
        """
        
        metrics = getLesionLevelDetectionMetrics( reference_image=self.reference, predicted_image=self.predicted_2of3TP_1FP )
        
        # Expected values
        TP = 2
        FN = 1
        FP = 1
        self.assertEqual( metrics.precision, precision(TP=TP,FP=FP) )
        self.assertEqual( metrics.recall, recall(TP=TP,FN=FN) )
        self.assertEqual( metrics.f1, f1(TP=TP,FP=FP,FN=FN) )
        self.assertEqual( metrics.TP, TP )
        self.assertEqual( metrics.FP, FP )      
        
if __name__ == '__main__':
    unittest.main()