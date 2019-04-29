So much wow!

### Data Creation

Call the `create_observations.py` script to create OGIP spectra thingyes from the stored DL3 data. It uses the configuration 
stored in `data_conf.yaml` for binning etc.


### Forward fit

Kinda works like this: give path to the folder contianing all the subdirectories for the telescope data. and a path under which to store the results.

```
    python fit_spectrum.py spectra/ results/fact/ fact --model_type wstat  --n_tune 2000 --n_samples 2000 

```

### Unfold Data

Call the `unfold_spectrum.py` script

```
    python unfold_spectrum.py data/ results/unfolding/veritas veritas

```