#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

_PETRainbowCMAP = matplotlib.colors.LinearSegmentedColormap(
    'PET-Rainbow',
    {
        u'blue': [(0.0, 0.0, 0.0),
                  (0.15, 0.6667, 0.6667),
                  (0.2, 0.8667, 0.8667),
                  (0.25, 0.8667, 0.8667),
                  (0.3, 0.8667, 0.8667),
                  (0.35, 0.8667, 0.8667),
                  (0.4, 0.8333, 0.8333),
                  (0.45, 0.0, 0.0),
                  (0.5, 0.0, 0.0),
                  (0.55, 0.0, 0.0),
                  (0.6, 0.0, 0.0),
                  (0.65, 0.0, 0.0),
                  (0.7, 0.0, 0.0),
                  (0.75, 0.0, 0.0),
                  (0.8, 0.0, 0.0),
                  (0.85, 0.1, 0.1),
                  (0.9, 0.2, 0.2),
                  (0.95, 0.3, 0.3),
                  (1.0, 1.0, 1.0)],
        u'green': [(0.0, 0.0, 0.0),
                   (0.15, 0.0, 0.0),
                   (0.2, 0.0, 0.0),
                   (0.25, 0.4667, 0.4667),
                   (0.3, 0.8, 0.8),
                   (0.35, 0.8667, 0.8667),
                   (0.4, 0.8667, 0.8667),
                   (0.45, 0.86, 0.86),
                   (0.5, 0.8633, 0.8633),
                   (0.55, 0.8667, 0.8667),
                   (0.6, 0.8667, 0.9667),
                   (0.65, 0.9667, 1.0),
                   (0.7, 1.0, 0.9333),
                   (0.75, 0.8, 0.8),
                   (0.8, 0.6, 0.6),
                   (0.85, 0.1, 0.1),
                   (0.9, 0.2, 0.2),
                   (0.95, 0.3, 0.3),
                   (1.0, 1.0, 1.0)],
        u'red': [(0.0, 0.0, 0.0),
                 (0.15, 0.0, 0.0),
                 (0.2, 0.0, 0.0),
                 (0.25, 0.0, 0.0),
                 (0.3, 0.0, 0.0),
                 (0.35, 0.0, 0.0),
                 (0.4, 0.0, 0.0),
                 (0.45, 0.0, 0.0),
                 (0.5, 0.0, 0.0),
                 (0.55, 0.0, 0.0),
                 (0.6, 0.0, 0.0),
                 (0.65, 0.7333, 0.7333),
                 (0.7, 0.9333, 0.9333),
                 (0.75, 1.0, 1.0),
                 (0.8, 1.0, 1.0),
                 (0.85, 1.0, 1.0),
                 (0.9, 0.8667, 0.8667),
                 (0.95, 0.8, 0.8),
                 (1.0, 1.0, 1.0)]
    },256)

def _contours_mask_slice(slice: np.ndarray,thickness: int=2, contour_position_outside=True):
    """
    Takes 2D binary image and computes its contour.
    Context: One wants to plot slices of segmentations 3D image.
    Args:
        slice (array): Numpy array with binary image.
    Returns:
        array : binary image containing contour of segmentation.
    """
    # Compute contours ...
    contours, _ = cv2.findContours(
        slice.astype(np.uint8).copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)
    # ... and put them in a binary image with contour
    mask = np.zeros(slice.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 1, thickness)

    # Make a mask with contour pixels in the segmentation masked
    if contour_position_outside:
        # (contours should be outside the segmentation)
        slice = np.logical_not(slice)
    return np.logical_not(np.logical_and(slice, mask))

def plot_img_and_mask( img,
                       mask,
                       ax=None,
                       # Image related arguments
                       vmin=None,
                       vmax=None,
                       cmap=_PETRainbowCMAP,
                       # Contour related arguments
                       line_color='r',
                       line_thickness=2,
                       overlay_mask=False,
                       contour_position_outside=True):
    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow( img, vmin=vmin, vmax=vmax, cmap=cmap )
    if overlay_mask:
        mask_ = mask.copy()
        mask_[ mask_ < 1 ] = np.nan
        ax.imshow( mask_, cmap=matplotlib.colors.ListedColormap([line_color]), alpha=0.3 )
    plot_mask( img=img,
               mask=mask,
               ax=ax,
               line_color=line_color,
               line_thickness=line_thickness,
               contour_position_outside=contour_position_outside
             )
    return ax

def plot_mask( img,
               mask,
               ax=None,
               line_color='r',
               line_thickness=2,
               contour_position_outside=True):
    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(
        np.ma.masked_where( _contours_mask_slice( mask, thickness=line_thickness, contour_position_outside=contour_position_outside ) , img ),
        cmap=matplotlib.colors.ListedColormap([line_color])
    )
    return ax
