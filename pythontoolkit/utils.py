#!/usr/bin/env python

import os

def listdir_nohidden(path):
    """List dir without hidden files
    
    Parameters
    ----------
    path : string
        Path to folder with files
    """
    return [f for f in os.listdir(path) if not f.startswith('.')]