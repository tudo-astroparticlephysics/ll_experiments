import os
from astropy.io import fits
from gammapy.spectrum import SpectrumObservationList


def load_spectrum_observations(input_dir, joint=False):
    """ Load the OGIP files and return a SpectrumObservationList
        SpectrumObservationList has already a method to read from a directory
        http://docs.gammapy.org/dev/api/gammapy.spectrum.SpectrumObservationList.html
    """

    if joint:
        spec_obs_list = SpectrumObservationList()
        # extend the list adding all the other SpectrumObservationList
        for n in {'magic', 'hess', 'fact', 'veritas'}:
            spectra_path = os.path.join(input_dir, n)
            spec_obs = SpectrumObservationList.read(spectra_path)
            spec_obs_list.extend(spec_obs)

    else:
        spec_obs_list = SpectrumObservationList.read(input_dir)

    hdu_list =  fits.open(os.path.join(input_dir, 'pha_stacked.fits') )
    h = hdu_list[1].header

    lo = h['LO_THRES']
    hi = h['HI_THRES']
    tel = h['TELESCOP']

    for obs in spec_obs_list:
        obs.meta['LO_THRES'] = lo
        obs.meta['HI_THRES'] = hi
        obs.meta['TELESCOP'] = tel

    return spec_obs_list, tel
