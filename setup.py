from setuptools import setup, find_packages

setup(
    name='pymc_spectrum',
    version='0.1.0',
    description='A tool to fit IACT 1D spectra',
    url='https://github.com/tudo-astroparticlephysics/ll_experiments',
    author='Kai Brügge',
    author_email='kai.bruegge@tu-dortmund.de',
    license='BEER',
    package_data={
        'resources/ascii': ['*.txt'],
        'resources/': ['*.txt'],
    },
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click',
        'h5py',
        'matplotlib>=3.0',
        'numexpr',
        'numpy',
        'pytest',
        'scipy',
        'tqdm',
        'pymc3==3.6',
        'gammapy==0.10',
        'astropy>=3.1.2',
        'ruamel.yaml>=0.15.0',
        'pydot===1.4.1',
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'pymc_fit_spectrum = pymc_spectrum.cli.fit_spectrum:main',
            'pymc_unfold_spectrum = pymc_spectrum.cli.unfold_spectrum:main',
            'pymc_create_observations = pymc_spectrum.cli.create_observations:main',
        ],
    }
)
