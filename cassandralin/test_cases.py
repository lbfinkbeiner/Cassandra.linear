from cassL import camb_interface as ci
from cassL import generate_emu_data as ged

import emulator_interface as ei
import numpy as np
import camb
import time

def Aletheia_to_cosmodict(index):
    return ci_to_cosmodict(ci.cosm.iloc[index])
    

def ci_to_cosmodict(c):
    return {
        "omega_b": c["ombh2"],
        "omega_cdm": c["omch2"],
        "ns": c["n_s"],
        "As": c["A_s"],
        "omega_nu": c["omnuh2"],
        "Omega_K": c["OmK"],
        "h": c["h"],
        "w0": c["w0"],
        "wa": c["wa"],
        "z": c["z"]
    }


def easy_comparisons(lhs, true, priors, k_axis):
    errors = np.empty(true.shape)
    predictions = np.empty(true.shape)

    for i in range(len(true)):
        print(i)
        this_denormalized_row = ged.denormalize_row(lhs[i], priors)
        this_cosmology = ged.build_cosmology(this_denormalized_row)
        this_cosmodict = ci_to_cosmodict(this_cosmology)
        
        try:
            this_intrpr, this_unc_intrpr = ei.get_Pk_interpolator(this_cosmodict)
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
        
    
