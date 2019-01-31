import os
from astropy.io import fits
from gammapy.spectrum import SpectrumObservationList
import glob


def load_joint_spectrum_observation(input_dir):
    """ Load the OGIP files and return a SpectrumObservationList
        SpectrumObservationList has already a method to read from a directory
        http://docs.gammapy.org/dev/api/gammapy.spectrum.SpectrumObservationList.html
    """
    if not (os.path.exists(input_dir) and os.listdir(input_dir)):
        raise ValueError('No files could be found under that path')

    spec_obs_list = SpectrumObservationList()
    # lo, hi = 0.08, 30
    lo, hi = get_fit_range_interval(input_dir)
    # extend the list adding all the other SpectrumObservationList
    for n in {'magic', 'hess', 'fact', 'veritas'}:
        spectra_path = os.path.join(input_dir, n)
        spec_obs = SpectrumObservationList.read(spectra_path)

        for obs in spec_obs:
            obs.meta['LO_RANGE'] = lo
            obs.meta['HI_RANGE'] = hi
            obs.meta['TELESCOP'] = n

        spec_obs_list.extend(spec_obs)

    return spec_obs_list, lo, hi

def get_fit_range_interval(input_dir):
    lo = []
    hi = []
    for n in {'magic', 'hess', 'fact', 'veritas'}:
        spectra_path = os.path.join(input_dir, n)
        l, h, _ = get_fit_settings(spectra_path)
        lo.append(l)
        hi.append(h)
    return min(lo), max(hi)

def load_spectrum_observations(input_dir):
    if not (os.path.exists(input_dir) and os.listdir(input_dir)):
        raise ValueError('No files could be found under that path')

    spec_obs_list = SpectrumObservationList.read(input_dir)

    lo, hi, tel = get_fit_settings(input_dir)

    for obs in spec_obs_list:
        obs.meta['LO_RANGE'] = lo
        obs.meta['HI_RANGE'] = hi
        obs.meta['TELESCOP'] = tel

    return spec_obs_list, lo, hi


def get_fit_settings(input_dir):
    ogip_file  = glob.glob(f'{input_dir}/pha*.fits')[0]

    with fits.open(ogip_file) as hdu_list:
        h = hdu_list[1].header

        lo = h['LO_RANGE']
        hi = h['HI_RANGE']
        tel = h['TELESCOP']

    return (lo, hi, tel)
