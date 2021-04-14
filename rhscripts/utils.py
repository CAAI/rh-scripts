#!/usr/bin/env python

import os
import itertools
import numpy as np


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