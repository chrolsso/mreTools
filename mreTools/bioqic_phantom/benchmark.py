import numpy as np

from mreTools.bioqic_phantom import gt

def calculateSwsDiffMasked(c_map):
    '''Calculates the difference (c_map - c_gt) between the ground truth and the given shear wave speed map in the regions of interest. Outside the regions of interest, the difference map will show 0 values.
    Expects c_map to be a 3D numpy array including shear wave speeds in m/s with dimensions (128, 80, 25).
    '''
    assert c_map.shape == (128, 80, 25), "The input array has to have dimensions (128, 80, 25)."

    c_gt = gt.getSwsGt()
    mask = gt.getMask()

    diff = c_map - c_gt
    diff[mask == 0] = 0

    return diff

def calculateMeanSwsPerRoi(c_map):
    '''Calculates the mean shear wave speed in m/s for each region of interest as well as the difference to the ground truth values.
    Expects c_map to be a 3D numpy array including shear wave speeds in m/s with dimensions (128, 80, 25).
    '''
    mask = gt.getMask()

    results = { 'roi1': {}, 'roi2': {}, 'roi3': {}, 'roi4': {}, 'matrix': {} }

    for roi in results.keys():
        mask_value = 0
        if roi == 'roi1':
            mask_value = 1
        elif roi == 'roi2':
            mask_value = 2
        elif roi == 'roi3':
            mask_value = 3
        elif roi == 'roi4':
            mask_value = 4
        elif roi == 'matrix':
            mask_value = 10

        mean_sws = np.mean(c_map[mask == mask_value])
        median_sws = np.median(c_map[mask == mask_value])

        results[roi]['mean'] = mean_sws
        results[roi]['median'] = median_sws
        results[roi]['mean_diff'] = np.abs(mean_sws - gt.getGtSwsForRoi(roi))
        results[roi]['median_diff'] = np.abs(median_sws - gt.getGtSwsForRoi(roi))

    return results

def calculateStorageModulusDiffMasked(G_prime_map):
    '''Calculates the difference (G_prime_map - G_prime_gt) between the ground truth and the given Storage modulus map in the regions of interest. Outside the regions of interest, the difference map will show 0 values.
    Expects G_prime_map to be a 3D numpy array including Storage modulus values in Pa with dimensions (128, 80, 25).
    '''
    assert G_prime_map.shape == (128, 80, 25), "The input array has to have dimensions (128, 80, 25)."

    G_prime_gt = gt.getStorageModulusGt()
    mask = gt.getMask()

    diff = G_prime_map - G_prime_gt
    diff[mask == 0] = 0

    return diff

def calculateMeanSwsPerRoi(c_map):
    '''Calculates the mean shear wave speed in m/s for each region of interest as well as the difference to the ground truth values.
    Expects c_map to be a 3D numpy array including shear wave speeds in m/s with dimensions (128, 80, 25).
    '''
    mask = gt.getMask()

    results = { 'roi1': {}, 'roi2': {}, 'roi3': {}, 'roi4': {}, 'matrix': {} }

    for roi in results.keys():
        mask_value = 0
        if roi == 'roi1':
            mask_value = 1
        elif roi == 'roi2':
            mask_value = 2
        elif roi == 'roi3':
            mask_value = 3
        elif roi == 'roi4':
            mask_value = 4
        elif roi == 'matrix':
            mask_value = 10

        mean_sws = np.mean(c_map[mask == mask_value])
        median_sws = np.median(c_map[mask == mask_value])

        results[roi]['mean'] = mean_sws
        results[roi]['median'] = median_sws
        results[roi]['mean_diff'] = np.abs(mean_sws - gt.getGtSwsForRoi(roi))
        results[roi]['median_diff'] = np.abs(median_sws - gt.getGtSwsForRoi(roi))

    return results