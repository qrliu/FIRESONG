#!/usr/bin/python
# Authors: Chris Tung
#          Ignacio Taboada
#

"""Example script that simulates a population of sources with a luminosity
    distribution that is dependent on redshift"""

# General imports
# from __future__ import division
import argparse
# Numpy / Scipy
import numpy as np
# Firesong code
from firesong.Evolution import get_LEvolution
from firesong.input_output import output_writer, print_config_LEGEND, get_outputdir
from firesong.sampling import InverseCDF


def legend_simulation(outputdir,
                      filename='LEGEND.out',
                      L_Evolution="HA2012BL",
                      zmax=10.,
                      bins=10000,
                      fluxnorm=1.44e-8,
                      index=2.28,
                      emin=1e4,
                      emax=1e7,
                      lmin=38,
                      lmax=48,
                      seed=None,
                      verbose=True):
    """
    Simulate a universe of neutrino sources with luminosity distribution 
        dependent on redshift

    Args:
        outputdir (str or None): path to write output. If None, return results
            without writing a file
        filename (str): name of the output file. 
        L_Evolution (str): Name of luminosity evolution model, or object of 
                           an evolution
        zmax (float, optional, default=10.): Farthest redshift to consider
        bins (int, optional, default=1000): Number of bins used when creating
            the redshift PDF
        fluxnorm (float, optional, default=1.44e-8): Normalization on the total
            astrophysical diffuse flux, E^2dPhi/dE. Units of GeV s^-1 sr^-1
        index (float, optional, default=2.13): Spectral index of diffuse flux
        emin (float, optional, default=1e4): Minimum neutrino energy in GeV
        emax (float, optional, default=1e7): Maximum neutrino energy in GeV
        lmin (float, optional, default=38): Minimum log10 luminosity in erg/s
        lmax (float, optional, default=38): Maximum log10 luminosity in erg/s
        seed (int or None, optional, default=None): random number seed
        verbose (bool, optional, default=True): print simulation paramaters
            if True else suppress printed output

    Returns:
        dict: keys contain simulation results, including the input params
            as well as the sources. Only returned if filename is None
    """

    if type(L_Evolution) == str:
        LE_model = get_LEvolution(L_Evolution, lmin, lmax)
    else:
        LE_model = L_Evolution
    
    if fluxnorm is not None:
        LE_model.DiffuseScaling(fluxnorm, index, emin, emax, zmax)

    N_sample = np.random.poisson(LE_model.Nsources(zmax))

    delta_gamma = 2 - index
    if verbose:
        print_config_LEGEND(L_Evolution, LE_model.lmin, LE_model.lmax, N_sample)

    ##################################################
    #        Simulation starts here
    ##################################################

    rng = np.random.RandomState(seed)

    # Prepare CDF for redshift generation
    redshift_bins = np.linspace(0.0005, zmax, bins)
    RedshiftPDF = [LE_model.RedshiftDistribution(z) for z in redshift_bins]
    invCDF = InverseCDF(redshift_bins, RedshiftPDF)

    # Prepare a luminosity CDF as a function of redshift
    luminosity_bins = np.linspace(LE_model.lmin, LE_model.lmax, 1000)
    l,z = np.meshgrid(luminosity_bins, redshift_bins)
    L_PDF = LE_model.LF(l,z)
    L_CDF = [InverseCDF(luminosity_bins, pdf) for pdf in L_PDF]
    
    if filename is not None:
        out = output_writer(outputdir, filename)
    else:
        results = {}

    # Generate redshift
    zs = invCDF(rng.uniform(low=0.0, high=1.0, size=N_sample))
    # Generate luminosity as function of z
    index_1 = np.searchsorted(redshift_bins, zs)
    lumis = [10**L_CDF[idx](rng.uniform(0.0, 1.0))[0] for idx in index_1]
    lumis = np.array(lumis)
    # Calculate the flux of each source
    fluxes = LE_model.Lumi2Flux(lumis, index, emin, emax, zs)
    # Random declination over the entire sky
    sinDecs = rng.uniform(-1, 1, size=N_sample)
    declins = np.degrees(np.arcsin(sinDecs))
    # Random ra over the sky
    ras = rng.uniform(0.,360., size=N_sample)
    TotalFlux = np.sum(fluxes)/4./np.pi

    # Write out
    if filename is not None:
        out.write(declins, ras, zs, fluxes)
        out.finish(TotalFlux)
    else:
        results['dec'] = declins
        results['ra'] = ras
        results['z'] = zs
        results['flux'] = fluxes

    # print before finish
    if verbose:
        print("Actual diffuse flux simulated :")
        log = "E^2 dNdE = {TotalFlux} (E/100 TeV)^({delta_gamma}) [GeV/cm^2.s.sr]"
        print(log.format(**locals()))

    if filename is None:
        return results


if __name__ == "__main__":
    outputdir = get_outputdir()

    # Process command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', action='store', dest='filename',
                        default='Legend.out', help='Output filename')
    parser.add_argument("--Levolution", action="store",
                        dest="Evolution", default='HA2012BL',
                        help="Source evolution options:  HA2012BL")
    parser.add_argument("--zmax", action="store", type=float,
                        dest="zmax", default=10.,
                        help="Highest redshift to be simulated")
    parser.add_argument("--index", action="store", dest='index',
                        type=float, default=2.19,
                        help="Spectral index of the outputflux")
    parser.add_argument("--lmin", action="store", dest="lmin",
                        type=float, default=41.5,
                        help="log10 of the minimum luminosity in erg/s")
    parser.add_argument("--lmax", action="store", dest="lmax",
                        type=float, default=41.5,
                        help="log10 of the maximum luminosity in erg/s")
    options = parser.parse_args()

    legend_simulation(outputdir,
                      filename=options.filename,
                      L_Evolution=options.Evolution,
                      zmax=options.zmax,
                      index=options.index,
                      lmin=options.lmin,
                      lmax=options.lmax)
