import numpy as np
from cassL import lhc
from cassL import generate_emu_data as ged
from cassL import train_emu as te
from cassL import camb_interface as ci
import os

#!! Matteo's code, which still needs to be gracefully incorporated
import cosmo_tools as brenda

import scipy
import warnings

# One of the benefits of distancing ourselves from the camb naming scheme is
# it makes debugging easier: we'll quickly understand whether there is a problem
# with the dev code or the user code.
DEFAULT_COSMOLOGY = {
    'omega_b': 0.022445,
    'omega_cdm': 0.120567,
    'ns': 0.96,
    'As': 2.127238e-9,
    'omega_K': 0,
    'omega_DE': 0.305888,
    'h': 0.67
}

DEFAULT_SIGMA12 = 0.82476394
# linear growth factor
DEFAULT_LGF = 0.7898639094999238

data_prefix = os.path.dirname(os.path.abspath(__file__)) + "/"

# Evolution parameter keys. If ANY of these appears in a cosmology dictionary,
# there had better not be a sigma12 value...
EV_PAR_KEYS = []

# Load the emulators that we need.
# sigma12 emu
sigma12_trainer = np.load(data_prefix + "emus/sigma12.cle")
# Massive-neutrino emu
nu_trainer = np.load(data_prefix + "emus/Hnu2.cle")
# Massless-neutrino emu ("zm" for "zero mass")
zm_trainer = np.load(data_prefix + "emus/Hz1.cle")

def contains_ev_par(dictionary):
    """
    Check if the cosmology specified by @dictionary contains a definition of
    an evolution parameter.
    """
    for ev_par_key in EV_PAR_KEYS:
        if ev_par_key in dictionary:
            return true
            
    return False

def estimate_sigma12(dictionary):
    """
    @dictionary should already be formatted according to the output of
        transcribe_cosmology.
    """
    
    # First, emulate sigma12 according to default evolution parameters
    return 23
    
    # Second, scale that sigma12 according to the growth factor and A_s

def dictionary_to_emu_vec(dictionary):
    """
    Turn an input cosmology into an input vector that the emulator understands
    and from which it can predict a power spectrum.
    """
    # Regardless of whether we use the massless or massive emu,
    # we need omega_b, omega_c, ns, and sigma12
    
    # We may want to allow the user to turn off error checking if a lot of
    # predictions need to be tested all at once...
    if "sigma12" in dictionary and contains_ev_par(dictionary):
        raise ValueError("sigma12 and at least one evolution parameter " + \
            "were simultaneously specified. If the desired sigma12 is " + \
            "already known, no evolution parameters should appear.")

    if "sigma12" not in dictionary:
        # We have to emulate this
        sigma12 = sigma12_trainer.p_emu.predict(base)
        # Now scale according to 
        dictionary["sigma12"] = estimate_sigma12(dictionary)
        return 23
    
    base = np.array([
        dictionary["omega_b"],
        dictionary["omega_cdm"],
        cosmology.pars["ns"],
        dictionary["sigma12"]
    ])
    
    if "omnu" == 0:
        return base
    else:
        extension = np.array([
            dictionary["A_s"],
            dictionary["omnu"]
        ])
        return np.append(base, extension)
    

def predict(dictionary):
    """
    This fn wraps the trainer object...
    
    It automatically returns a prediction and the estimated uncertainty on that
    prediction. I still don't really know why we re-invented the wheel, when
    the GPR object itself gives an uncertainty. But let's see...
    """
    # Can I use an API that I can't necessarily see?
    # i.e. does it work to access trainer fn.s if cassL-dev isn't
    # installed?
    
    # If the user has not provided a sigma12 value, we need to compute it from
    # the evolution parameters. If there is a sigma12 value, we need to 
    # complain in the presence of evolution parameters...
 
    if dictionary["omega_nu"] == 0:
        raise NotImplementedError("activate massless-neutrino emu")
        
        # Don't forget to normalize the x first!!
        test_predictions[i] = self.p_emu.predict(X_test[i])
    else
        raise NotImplementedError("activate massive-neutrino emu")
        
    # Now apply some rescaling to the result based on the provided evolution
    # parameters...
    return 23
    

def prior_file_to_array(prior_name="COMET_with_nu"):
    """
    !
    """
    param_ranges = None

    prior_file = data_prefix + "priors/" + prior_name + ".txt"
    
    with open(prior_file, 'r') as file:
        lines = file.readlines()
        key = None
        for line in lines:
            if line[0] != "$":
                bounds = line.split(",")
                # Extra layer of square brackets so that np.append works
                bounds = np.array([[float(bounds[0]), float(bounds[1])]])
                
                if param_ranges is None:
                    param_ranges = bounds
                else:
                    param_ranges = np.append(param_ranges, bounds, axis=0)

    return param_ranges


def get_data_dict(scenario_name):
    #! WATCH OUT! THIS FUNCTION ASSUMES MASSIVE NEUTRINOS ALWAYS

    # This will return a dictionary which the new iteration of
    # build_and_test_emulator will be able to expand into all of the info
    # necessary to build an emulator.
    
    # e.g. emu_name is Hnu2_5k_knockoff
    
    scenario = get_scenario(scenario_name)
    
    # This function will have to be expanded dramatically once we implement
    # the scenario structure
    directory = "data_sets/" + scenario_name + "/"
    data_dict = {"emu_name": scenario_name}

    X_train = np.load(directory + "lhc_train_final.npy", allow_pickle=False)
    data_dict["X_train"] = X_train

    Y_train = np.load(directory + "samples_train.npy", allow_pickle=False)
    data_dict["Y_train"] = Y_train

    if scenario["same_test_set"] is not None:
        directory = "data_sets/" + scenario["same_test_set"] + "/"

    X_test = np.load(directory + "lhc_test_final.npy", allow_pickle=False)
    data_dict["X_test"] = X_test
    
    Y_test = np.load(directory + "samples_test.npy", allow_pickle=False)
    data_dict["Y_test"] = Y_test
    
    data_dict["priors"] = prior_file_to_array(scenario["priors"])

    return data_dict


def E2(OmM_0, OmK_0, OmDE_0, z):
    return OmM_0 * (1 + z) ** 3 + OmK_0 * (1 + z) ** 2 + OmDE_0


def linear_growth_factor(OmM_0, OmK_0, OmDE_0, z):
    def integrand(zprime):
        return (1 + zprime) / np.power(E2(OmM_0, OmK_0, OmDE_0, zprime), 1.5)

    coefficient = 2.5 * OmM_0 * np.sqrt(E2(OmM_0, OmK_0, OmDE_0, z))
    integral = scipy.integrate.quad(integrand, z, np.inf)[0]
    return coefficient * integral


spec_conflict_message = "Do not attempt to simultaneously set curvature, " + \
    "dark energy, and the Hubble parameter. Set two of the three, and " + \
    "Cassandra-Linear will automatically handle the third."

# "Handle" is kind of a misleading term; if the user specifies a physical
# density, we won't bother with fractions at all.
doubly_defined_message = "Do not simultaneously specify the physical and " + \
    "fractional density in {}. Specify one, and " + \
    "Cassandra-Linear will automatically handle the other."
    
missing_h_message = "A fractional density parameter was specified, but no " + \
    "value of 'h' was provided."
    
missing_shape_message = "The value of {} was not provided. This is an " + \
    "emulated shape parameter and is required. Setting to the Planck " + \
    "best-fit value..."

def transcribe_cosmology(**kwargs):
    """
    Turn a set of arguments into a complete Cosmology object. Cosmology
    objects follow a particular format for compatibility with the fn.s in
    this script (and in Brendalib) that return various power spectra power.
    
    This fn. thoroughly error-checks the arguments to verify that they represent
    a consistent and complete cosmology. Mostly, completeness is not a problem
    for the user, as missing parameters are generally inferred from the default 
    cosmology. However, for example, this fn will complain if the user attempts
    to specify fractional density parameters without specifying the value of the 
    Hubble parameter.
    
    After this verification, the fn converts given quantities to desired
    quantities. For example, the code in this script primarily uses physical 
    densities, so fractional densities will be converted.
    
    Possible parameters:
    omB: float
        Physical density in baryons
    OmB: float
        Fractional density in baryons
    
    omC: float
        Physical density in cold dark matter
    OmC: float
        Fractional density in cold dark matter
    
    omDE: float
        Physical density in dark energy
    OmDE: float
        Fractional density in dark energy
        
    omnu: float
        Physical density in neutrinos
    Omnu: float
        Fractional density in neutrinos
    
    omK: float
        Physical density in curvature
    OmK: float
        Fractional density in curvature
    
    h: float
        dimensionless Hubble parameter
    H0: float
        Hubble parameter in km / s / Mpc
        
    ns: float
        Spectral index of the primordial power spectrum
    
    As: float
        Scalar mode amplitude of the primordial power spectrum
        
    !!!
    z: float
        redshift. THIS PROBABLY DOESN'T BELONG IN THE COSMOLOGY DICTIONARY.
        Maybe we should leave redshift as a separate input in the various fn.s
        of this script?
    """
    # Instead of directly building a brenda Cosmology, we use this temporary
    # dictionary; it helps to keep track of values that may need to be
    # converted or inferred from others.
    conversions = {}
    
    if "w" in kwargs:
        conversions["w0"] = kwargs["w"]
    if "w0" in kwargs:
        conversions["w0"] = kwargs["w0"]
    if "wa" in kwargs:
        conversions["wa"] = kwargs["wa"]

    # Make sure that no parameters are doubly-defined
    if "omega_b" in kwargs and "Omega_b" in kwargs:
        raise ValueError(str.format(doubly_defined_message, "baryons"))
    if "omega_cdm" in kwargs and "Omega_cdm" in kwargs:
        raise ValueError(str.format(doubly_defined_message,
                                      "cold dark matter"))
    if "omega_DE" in kwargs and "Omega_DE" in kwargs:
        raise ValueError(str.format(doubly_defined_message, "dark energy"))
    if "omega_K" in kwargs and "Omega_K" in kwargs:
        raise ValueError(str.format(doubly_defined_message, "curvature"))
    if "h" in kwargs and "H0" in kwargs:
        raise ValueError("Do not specify h and H0 simultaneously. Specify " + \
            "one, and Cassandra-Linear will automatically handle the other.")

    # Make sure at most two of the three are defined: h, omega_curv, omega_DE
    if "h" in kwargs or "H0" in kwargs:
        if "Omega_DE" in kwargs or "omega_DE" in kwargs:
            if "Omega_K" in kwargs or "omk" in kwargs:
                raise ValueError(spec_conflict_message)

    # If h is present, set it right away, so that we can begin converting
    # fractional densities.
    if "h" in kwargs:
        conversions["h"] = kwargs["h"]
    elif "H0" in kwargs:
        conversions["h"] = kwargs["H0"] / 100

    # Make sure that h is present, in the event that fractionals are given
    fractional_keys = ["Omega_b", "Omega_cdm", "Omega_DE", "Omega_K", "Omnu"]
    physical_keys = ["omega_b", "omega_cdm", "omega_DE", "omega_K", "omnu"]
    
    for i in range(len(fractional_keys)):
        frac_key = fractional_keys[i]
        if frac_key in kwargs:
            if "h" not in conversions:
                raise ValueError(missing_h_message)
            phys_key = physical_keys[i]
            conversions[phys_key] = kwargs[frac_key] * conversions["h"] ** 2
    
    # Nothing else requires such conversions, so add the remaining values
    # directly to the working dictionary.
    for key, value in kwargs.items():
        if key not in fractional_keys:
            conversions[key] = value
    
    # Now complain about missing entries. We have to fill in missing densities
    # immediately because they will be used to find h.
    if "omega_b" not in kwargs:
        warnings.warn(str.format(missing_shape_message, "omega_b"))
        conversions["omega_b"] = DEFAULT_COSMOLOGY["omega_b"]

    # Ditto with cold dark matter.
    if "omega_cdm" not in kwargs:
        warnings.warn(str.format(missing_shape_message, "omega_cdm"))
        conversions["omega_cdm"] = DEFAULT_COSMOLOGY["omega_cdm"]

    # Ditto with neutrinos.
    if "omnu" not in kwargs:
        warnings.warn("The value of 'omnu' was not provided. Assuming " + \
                      "massless neutrinos..."))
        conversions["omnu"] = DEFAULT_COSMOLOGY["omnu"]

    # Ditto with the spectral index.
    if "ns" not in kwargs:
        warnings.warn(str.format(missing_shape_message, "ns"))
        conversions["ns"] = DEFAULT_COSMOLOGY["ns"]
        
    if "z" not in kwargs:
        warnings.warn("No redshift given. Using z=0...")
        conversions["z"] = 0

    omM = conversions["omega_b"] + conversions["omega_cdm"] + conversions["omnu"]

    # The question is, when omK is not specified, should we immediately set it
    # to default, or immediately set h to default and back-calculate curvature?
    if "omega_DE" in conversions and "omega_K" not in conversions:
        if "h" not in conversions:
            # use default value for h
            conversions["omega_K"] = DEFAULT_COSMOLOGY["omega_K"]
        else:
            conversions["omega_K"] = \
                conversions["h"] ** 2 - omM - conversions["omega_DE"]

    # Analogous block for dark energy
    if "omega_K" in conversions and "omega_DE" not in conversions:
        if "h" not in conversions:
            # use default value for h
            conversions["h"] = DEFAULT_COSMOLOGY["h"]
        else:
            conversions["omega_DE"] = \
                conversions["h"] ** 2 - omM - conversions["omega_K"]

    # Fill in default values for density parameters, because we need these to
    # compute h
    if "omega_K" not in conversions:
        conversions["omega_K"] = DEFAULT_COSMOLOGY["omega_K"]

    # If omDE was never given, there's no point in calculating h
    if "omega_DE" not in conversions:
        conversions["h"] = DEFAULT_COSMOLOGY["h"]
        conversions["omega_DE"] = DEFAULT_COSMOLOGY["omega_DE"]

    # If h wasn't given, compute it now that we have all of the physical
    # densities.
    if "h" not in conversions:
        DEFAULT_COSMOLOGY["h"] = np.sqrt(omM + omDE + omK)
        
    raise 1 / 0    
        
    # package it up for brenda
    if "As" not in conversions:
        conversions["As"] = DEFAULT_COSMOLOGY["As"]
    
    cosmology = brenda.Cosmology()  
    conversions["de_model"] = "w0wa"
    # The omegaK field will be ignored, but remembered through h
    cosmology.pars = conversions
        
    return cosmology

def scale_sigma12(**kwargs):
    """
    Ideally, the user wouldn't use this function, it would automatically be
    called under the hood in the event that the user attempts to specify
    evolution parameters in addition to the mandatory shape params.

    Preferentially uses a specified 'h' to define omega_DE while leaving omega_K
    as the default value.
    :param kwargs:
    :return:
    """
    conversions = transcribe_cosmology(kwargs)

    omM = conversions["omega_b"] + conversions["omega_cdm"]
    OmM = omM / conversions["h"] ** 2
    OmK = conversions["omega_K"] / conversions["h"] ** 2
    OmDE = conversions["omega_DE"] / conversions["h"] ** 2
    
    LGF = linear_growth_factor(OmM, OmK, OmDE, conversions["z"])
    
    #! Should we throw an error if A_s is specified without omega_nu?
    #! I don't think so...
    As_ratio = 1
    if "A_s" in conversions:
        As_ratio = conversions["A_s"] / DEFAULT_COSMOLOGY["A_s"]
    
    return DEFAULT_SIGMA12 * LGF / DEFAULT_LGF * np.sqrt(As_ratio)

