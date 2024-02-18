from cassL import camb_interface as ci
from cassL import generate_emu_data as ged
from cassL import utils

import emulator_interface as ei
import numpy as np
import camb
import time

def Aletheia_to_cosmodict(index):
    return ci_to_cosmodict(ci.cosm.iloc[index])
    

def ci_to_cosmodict(c):
    base = {
        "omega_b": c["ombh2"],
        "omega_cdm": c["omch2"],
        "ns": c["n_s"],
        "As": c["A_s"],
        "omega_nu": c["omnuh2"],
        "Omega_K": c["OmK"],
        "h": c["h"],
        "w0": c["w0"],
        "wa": c["wa"],
    }
    if "z" in c:
        base["z"] = c["z"]
    else:
        base["z"] = 0

    return base

def toss_ev_pars(c):
    alt_cosm = ci.default_cosmology()
    alt_cosm["ombh2"] = c["ombh2"]
    alt_cosm["omch2"] = c["omch2"]
    alt_cosm["n_s"] = c["n_s"]
    alt_cosm["A_s"] = c["A_s"]
    alt_cosm["z"] = 0
    alt_cosm = ci.specify_neutrino_mass(alt_cosm, c["omnuh2"], 1)
    return alt_cosm

def easy_comparisons_sigma12(lhs, priors, k_axis):
    perc_errors = []
    true = []
    predictions = []

    for i in range(len(lhs)):
        print(i)
        this_denormalized_row = ged.denormalize_row(lhs[i], priors)
        this_cosmology = ged.build_cosmology(this_denormalized_row)
        print(this_cosmology)
        cd = ci_to_cosmodict(this_cosmology)
        cd = ei.convert_fractional_densities(cd)
        cd = ei.fill_in_defaults(cd)
        this_brendac = ei.transcribe_cosmology(cd)
        print(this_brendac.pars)

        try:
            this_true = ci.evaluate_sigma12(this_cosmology, 
                    redshifts=[this_cosmology["z"]])[0]
            this_pred = ei.add_sigma12(this_brendac).pars['sigma12']
            this_percerr = utils.percent_error(this_true, this_pred)

            true.append(this_true)
            predictions.append(this_pred)
            perc_errors.append(this_percerr)
        except ValueError:
            true.append(np.nan)
            predictions.append(np.nan)
            perc_errors.append(np.nan)

    return perc_errors, true, predictions


def easy_comparisons(lhs, true, priors, k_axis):
    errors = np.empty(true.shape)
    predictions = np.empty(true.shape)

    for i in range(len(true)):
        print(i)
        this_denormalized_row = ged.denormalize_row(lhs[i], priors)
        this_cosmology = ged.build_cosmology(this_denormalized_row)
        this_cosmodict = ci_to_cosmodict(this_cosmology)
        
        try:
            this_intrpr, this_unc_intrpr = \
                ei.get_Pk_interpolator(this_cosmodict)
            this_prediction = this_intrpr(k_axis)
            predictions[i] = this_prediction
            errors[i] = this_prediction - true[i]
        except ValueError:
            errors[i][:] = np.nan

    return errors, predictions


def time_simple():
    cosmo_dict = Aletheia_to_cosmodict(4)
    ei.error_check_cosmology(cosmo_dict)

    cosmo_dict = ei.convert_fractional_densities(cosmo_dict)
    cosmo_dict = ei.fill_in_defaults(cosmo_dict)

    cosmology = ei.transcribe_cosmology(cosmo_dict)

    if "sigma12" not in cosmology.pars:
        cosmology = ei.add_sigma12(cosmology)
    else:
        if not ei.within_prior(cosmology.pars["sigma12"], 3):
            raise ValueError(str.format(ei.OUT_OF_BOUNDS_MSG, "sigma12"))

    emu_vector = ei.cosmology_to_emu_vec(cosmology)

    start_time = time.time()
    # Again, we have to de-nest
    if len(emu_vector) == 6:  # massive neutrinos
        ei.NU_TRAINER.p_emu.predict(emu_vector)[0], \
            ei.NU_TRAINER.delta_emu.predict(emu_vector)[0]
    elif len(emu_vector) == 4:  # massless neutrinos
        ei.ZM_TRAINER.p_emu.predict(emu_vector)[0], \
            ei.ZM_TRAINER.delta_emu.predict(emu_vector)[0]

    return time.time() - start_time

def time_package():
    start_time = time.time()
    Pk0, unc0 = ei.get_Pk_interpolator(Aletheia_to_cosmodict(4))
    Pk0(3e-3)
    return time.time() - start_time
    
def time_CAMB_package():
    """
    !!!
    We'll need separate time tests for the case where neutrinos are massive--
    I'm almost certain that this case requires more compute time in CAMB.
    """
    start_time = time.time()
    ci.evaluate_cosmology(ci.specify_neutrino_mass(ci.cosm.iloc[4], 0))
    return time.time() - start_time
    
def time_CAMB_simple():
    k_points = len(ei.K_AXIS)
    cosmology = ci.specify_neutrino_mass(ci.cosm.iloc[4], 0)
    redshifts = np.array([0])

    if not isinstance(redshifts, list) and \
        not isinstance(redshifts, np.ndarray):
        raise TypeError("If you want to use a single redshift, you must " + \
            "still nest it in an array.")

    pars = ci.input_cosmology(cosmology)

    ci.apply_universal_output_settings(pars)

    start_time = time.time()

    # To change the the extent of the k-axis, change the following line as
    # well as the "get_matter_power_spectrum" call.
    pars.set_matter_power(redshifts=redshifts, kmax=10.0 / pars.h,
                          nonlinear=False)
                          
    smpt = time.time()
                          
    results = camb.get_results(pars)

    grt = time.time()

    sigma12 = results.get_sigmaR(12, hubble_units=False)

    s12t = time.time()

    # In some cursory tests, the accurate_massive_neutrino_transfers
    # flag did not appear to significantly alter the outcome.

    k, z, p = results.get_matter_power_spectrum(
        minkh=1e-4 / pars.h, maxkh=10.0 / pars.h, npoints=k_points,
        var1='delta_nonu', var2='delta_nonu'
    )
    
    end = time.time()
    
    return smpt - start_time, grt - smpt, s12t - grt, end - s12t, \
        end - start_time
        
    
