#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:47:52 2021

@author: claes
"""

from skimage import measure
import collections
import numpy as np

def getLesionLevelDetectionMetrics( reference_image: np.ndarray, predicted_image: np.ndarray ) -> collections.namedtuple:    
    """
    
    Lesion-level detection metrics

    Parameters
    ----------
    reference_image : np.ndarray
        Reference image of zeros (background) and ROIs (above zero).
    predicted_image : np.ndarray
        New image of zeros (background) and ROIs (above zero).

    Returns
    -------
    metrics
        Named tuple of metrics. Get e.g. TP by calling metrics.TP.

    """
    
    predicted_clusters = measure.label( predicted_image, background=0 )
    true_clusters = measure.label( reference_image, background=0 )
    overlap = np.multiply(true_clusters, predicted_image) 
    
    numTrueClusters = np.max(true_clusters)
    numPredClusters = np.max(predicted_clusters)
    
    TP = len(np.unique(overlap)) - 1 # 1 for BG
    FP = numPredClusters - TP
    
    recall = 0 if numTrueClusters == 0 else TP / numTrueClusters
    precision = 0 if numPredClusters == 0  else TP  / numPredClusters
    f1 = any([precision,recall]) and 2*(precision*recall)/(precision+recall) or 0
    
    Metrics = collections.namedtuple("Metrics", ["precision", "recall", "f1", "TP", "FP"])
    return Metrics(precision=precision, recall=recall, f1=f1, TP=TP, FP=FP)

def getLesionLevelDetectionMetricsV2( reference_image: np.ndarray, predicted_image: np.ndarray ) -> collections.namedtuple:    
    """
    
    Lesion-level detection metrics
    
    V2 runs through all predicted lesions first, and counts the TP and FP. 
    Then all ref-lesions are examined for FN. Compared to V1, this ensures 
    that FP >= 0 and sensitivity<=1.

    Parameters
    ----------
    reference_image : np.ndarray
        Reference image of zeros (background) and ROIs (above zero).
    predicted_image : np.ndarray
        New image of zeros (background) and ROIs (above zero).

    Returns
    -------
    metrics
        Named tuple of metrics. Get e.g. TP by calling metrics.TP.

    """
    
    predicted_clusters = measure.label( predicted_image, background=0 )
    true_clusters = measure.label( reference_image, background=0 )
    
    TP,FP,FN = 0,0,0
    for ind in range(predicted_clusters.max()):
        lp = np.zeros(reference_image.shape)
        lp[ predicted_clusters == ind+1 ] = 1.0
        if np.sum( lp * reference_image ) > 0:
            TP+=1
        else:
            FP+=1
    for ind in range(true_clusters.max()):
        lr = np.zeros(reference_image.shape)
        lr[ true_clusters == ind+1 ] = 1.0
        if np.sum( lr * predicted_image ) == 0:
            FN+=1
    
    P = FN+TP
    numPredClusters = TP+FP
    
    recall = 0 if P == 0 else TP / P
    precision = 0 if numPredClusters == 0  else TP  / numPredClusters
    f1 = any([precision,recall]) and 2*(precision*recall)/(precision+recall) or 0
    
    Metrics = collections.namedtuple("Metrics", ["precision", "recall", "f1", "TP", "FP", "FN"])
    return Metrics(precision=precision, recall=recall, f1=f1, TP=TP, FP=FP, FN=FN)