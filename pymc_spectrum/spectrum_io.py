import os
from astropy.io import fits
from gammapy.spectrum import SpectrumObservationList
import glob


def get_fit_range_interval(input_dir):
    '''
    Returns the minimun and maximum of all fit ranges in the
    OGIP files in the subdirectories with names
    'magic', 'hess', 'fact', 'veritas'.
    '''
    lo = []
    hi = []
    for n in ['magic', 'hess', 'fact', 'veritas']:
        spectra_path = os.path.join(input_dir, n)
        l, h, _ = get_fit_settings(spectra_path)
        lo.append(l)
        hi.append(h)
    return min(lo), max(hi)


def load_spectrum_observations(input_dir):
    '''
    Create a SpectrumObservationList containing all observations stored
    within 'input_dir'

    Returns
    SpectrumObservationList, low_fit_range, high_fit_range
    '''
    if not (os.path.exists(input_dir) and os.listdir(input_dir)):
        raise ValueError(f'No files could be found under that path: {input_dir}')

    spec_obs_list = SpectrumObservationList.read(input_dir)
    
    lo, hi, tel = get_fit_settings(input_dir)

    for obs in spec_obs_list:
        obs.meta['LO_RANGE'] = lo
        obs.meta['HI_RANGE'] = hi
        obs.meta['TELESCOP'] = tel

    return spec_obs_list, lo, hi


def get_fit_settings(input_dir):
    '''
    Reads the header of the ogip file to get the fit range.
    '''
    ogip_file = glob.glob(f'{input_dir}/pha*.fits')[0]

    with fits.open(ogip_file) as hdu_list:
        h = hdu_list[1].header

        lo = h['LO_RANGE']
        hi = h['HI_RANGE']
        tel = h['TELESCOP']

    return (lo, hi, tel)
