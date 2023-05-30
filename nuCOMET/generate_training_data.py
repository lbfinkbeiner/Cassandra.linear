# It seems like tkgrid.py might have been intended to help me with this,
# but I don't know how to use it

# Short-cut:
# Documents\GitHub\Master\CAKE21\modded_repo

import sys, platform, os
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

import camb
from camb import model, initialpower, get_matter_power_interpolator
import pandas as pd
import re

import camb_interface as ci

cosm = pd.read_csv("cosmologies.dat", sep='\s+')
model0 = cosm.loc[0]

''' AndreaP thinks that npoints=300 should be a good balance of accuracy and
computability for our LH.'''
NPOINTS = 300

import sys, traceback
import copy as cp

# These values help with the following function.
# However, neither of these belongs here, we should find a different home.
disregard_keys = ["OmB", "OmC", "OmM", "z(4)", "z(3)", "z(2)", "z(1)", "z(0)",
    "Lbox", "sigma8", "Name", "nnu_massive", "EOmDE"]

def print_cosmology(cosmology):
    for key in cosmology.keys():
        if key not in disregard_keys:
            print(key, cosmology[key])
    print()

def build_cosmology(om_b_in, om_c_in, ns_in, om_nu_in, sigma12_in, As_in):
    # Use Aletheia model 0 as a base
    cosmology = cp.deepcopy(ci.cosm.iloc[0])
    
    cosmology["ombh2"] = om_b_in
    cosmology["omch2"] = om_c_in
    cosmology["n_s"] = ns_in
    # Incomplete
    cosmology["sigma12"] = sigma12_in
    cosmology["A_s"] = As_in

    ''' Actually the last argument is not really important and is indeed just
        the default value. I'm writing this out explicitly because we're still
        in the debugging phase and so my code should always err on the verbose
        side.'''
    nnu_massive = 0 if om_nu_in == 0 else 1

    return ci.specify_neutrino_mass(cosmology, om_nu_in,
        nnu_massive_in=nnu_massive)

def fill_hypercube(parameter_values, standard_k_axis, cell_range=None,
    samples=None, write_period=None):
    """
    @parameter_values: this should be a list of tuples to
        evaluate kp at.

    @cell_range adjust this value in order to pick up from where you
        left off, and to run this method in saveable chunks.
    """
    if cell_range is None:
        cell_range = range(len(parameter_values))
    if samples is None:
        samples = np.zeros((len(parameter_values), NPOINTS))

    unwritten_cells = 0
    for i in cell_range:
        #print(i, "computation initiated")
        config = parameter_values[i]
        #print(config, "\n", config[4])
        p = None
        #try:
            #print("beginning p-spectrum computation")
        cosmology = build_cosmology(config[0], config[1], config[2],
                config[4], config[3], config[5])
        print_cosmology(cosmology)
        p = kp(cosmology, standard_k_axis)
            #print("p-spectrum computation complete!")
        #except ValueError:
        ''' Don't let unreasonable sigma12 values crash the program; ignore
        them for now. It's not clear to me why unreasonable sigma12 values
        sometimes (albeit quite rarely) raise ValueErrors. One would think
        that that situation would be adequately handled by the h=0.01 check
        in kp.
        '''
        #    traceback.print_exc(limit=1, file=sys.stdout)
        #except Exception: 
        #    traceback.print_exc(limit=1, file=sys.stdout)
        
        samples[i] = p
        
        print(i, "complete")
        unwritten_cells += 1
        if write_period is not None and unwritten_cells >= write_period:
            np.save("samples_backup_i" + str(i) + ".npy", samples,
                allow_pickle=True)
            unwritten_cells = 0
    return samples

def kp(cosmology, standard_k_axis):
    """
    Returns the scale axis and power spectrum in Mpc units

    @h_in=0.67 starts out with the model 0 default for Aletheia, and we
        will decrease it if we cannot get the desired sigma12 with a
        nonnegative redshift.
    """
    #print("Trying with h", cosmology['h'])
    #print("min z", min(_redshifts), "max z", max(_redshifts))
    
    # This allows us to roughly find the z corresponding to the sigma12 that we
    # want.
    _redshifts=np.flip(np.linspace(0, 10, 150))
    
    _, _, _, list_sigma12 = ci.kzps(cosmology, _redshifts,
        fancy_neutrinos=False, k_points=NPOINTS)

    # debug block
    
    #print(list_s12)
    if False:
        import matplotlib.pyplot as plt
        # Original intersection problem we're trying to solve
        plt.plot(_redshifts, list_sigma12);
        plt.axhline(cosmology["sigma12"], c="black")
        plt.title("$\sigma_{12}$ vs. $z$")
        plt.ylabel("$\sigma_{12}$")
        plt.xlabel("$z$")
        plt.show()
        # Now it's a zero-finding problem
        plt.plot(_redshifts, list_sigma12 - cosmology["sigma12"]);
        plt.axhline(0, c="black")
        plt.title("$\sigma_{12} - \sigma^{\mathrm{goal}}_{12}$ vs. $z$")
        plt.xlabel("$z$")
        plt.ylabel("$\sigma_{12} - \sigma^{\mathrm{goal}}_{12}$")
        plt.show()
    
    list_sigma12 -= cosmology["sigma12"] # now it's a zero-finding problem
    
    # remember that list_s12[0] corresponds to the highest value z
    if list_sigma12[len(list_sigma12) - 1] < 0:
        ''' we need to start playing with h.
        To save on computation, let's check if even the minimum allowed value
        rescues the problem.
        '''
        #print("We need to move h")
        #print(cosmology)
        if cosmology['h'] <= 0.1:
            print("This cell is hopeless. Here are the details:")
            print(cosmology)
            return None

        cosmology['h'] -= 0.1
        return kp(cosmology, standard_k_axis)

    z_step = _redshifts[0] - _redshifts[1]
    interpolator = interp1d(np.flip(_redshifts), np.flip(list_sigma12),
        kind='cubic')
        
    # Newton's method requires that I already almost know the answer, so it's
    # poorly suited to our problem. This generic root finder works better.
    z_best = root_scalar(interpolator,
        bracket=(np.min(_redshifts), np.max(_redshifts))).root

    p = np.zeros(len(standard_k_axis))

    if cosmology['h'] == model0['h']: # if we haven't touched h,
        # we don't need to interpolate.
        _, _, p, actual_sigma12 = ci.kzps(cosmology,
            redshifts=np.array([z_best]), fancy_neutrinos=False,
            k_points=NPOINTS) 
       
    else: # it's time to interpolate
        print("We had to move h to", np.around(cosmology['h'], 3))
        PK = ci.kzps_interpolator(cosmology, redshifts=_redshifts,
            fancy_neutrinos=False, z_points=150,
            kmax=max(standard_k_axis), hubble_units=False)

        p = PK.P(z_best, standard_k_axis)

    if len(p) == 1:
        p = p[0] 

    # We don't need to return k because we take for granted that all
    # runs will have the same k axis.

    # This one's for Andrea. Don't delete it until you've collected some
    # results.
    print("Redshift used:", z_best)

    return p, actual_sigma12, z_best
