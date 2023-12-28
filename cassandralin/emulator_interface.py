import numpy as np
from cassL import lhc
from cassL import generate_emu_data as ged
from cassL import train_emu as te
from cassL import camb_interface as ci
import os

import copy as cp

#!! Matteo's code, which still needs to be gracefully incorporated
import cosmo_tools as brenda

import scipy
import warnings

DEFAULT_COSMOLOGY = {
    'omega_b': 0.022445,
    'omega_cdm': 0.120567,
    'ns': 0.96,
    'As': 2.127238e-9,
    'omega_K': 0,
    'omega_DE': 0.305888,
    'omega_nu': 0,
    'h': 0.67,
    'sigma12': 0.82476394,
    'LGF': 0.7898639094999238
}

data_prefix = os.path.dirname(os.path.abspath(__file__)) + "/"

# Evolution parameter keys. If ANY of these appears in a cosmology dictionary,
# there had better not be a sigma12 value...
EV_PAR_KEYS = []

# Load the emulators that we need.
# sigma12 emu
sigma12_trainer = np.load(data_prefix + "emus/sigma12.cle", allow_pickle=True)
# Massive-neutrino emu
nu_trainer = np.load(data_prefix + "emus/Hnu2.cle", allow_pickle=True)
# Massless-neutrino emu ("zm" for "zero mass")
zm_trainer = np.load(data_prefix + "emus/Hz1.cle", allow_pickle=True)

def prior_file_to_array(prior_name="COMET"):
    """
    Read a prior file into a NumPy array.
    
    Prior files are a feature from the developer tools; they control the
    building of the data sets over which the emulators train. This function,
    which originally appears there, has been copied here  only for the purpose
    of providing more accessible error messages when the user inputs a value
    outside of the acceptable priors.
    
    In other words, users only interested in this interface will have no
    reason to call this function outside of its automatic invocation.
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

PRIORS = prior_file_to_array("COMET")

def within_prior(value, index):
    """
    Check if a given value falls within the associated priors.
    
    @value the parameter value to be tested
    @index the index of PRIORS which corresponds to that parameter. For 
        example, if @value were a configuration of the spectral index. We would
        call within_prior(value, 2).
    """
    return value >= PRIORS[index][0] and value <= PRIORS[index][1]

def contains_ev_par(cosmo_dict):
    """
    Check if the cosmology specified by @cosmo_dict contains a definition of
    an evolution parameter.
    """
    for ev_par_key in EV_PAR_KEYS:
        if ev_par_key in cosmo_dict:
            return true
            
    return False

ambiguous_sigma12_msg = "sigma12 and at least one evolution parameter " + \
    "were simultaneously specified. If the desired sigma12 is already " + \
    "known, no evolution parameters should appear."

spec_conflict_msg = "Do not attempt to simultaneously set curvature, " + \
    "dark energy, and the Hubble parameter. Set two of the three, and " + \
    "Cassandra-Linear will automatically handle the third."

doubly_defined_msg = "Do not simultaneously specify the physical and " + \
    "fractional density in {}. Specify one, and " + \
    "Cassandra-Linear will automatically handle the other."
    
out_of_bounds_msg = "The given value for {} falls outside of the range " + \
    "over which the emulators were trained. Try a less extreme value."

def error_check_cosmology(cosmo_dict):
    """
    Provide clear error messages for some, but not all, cases of ambiguity or
    inconsistency in the input parameters.
    1. Both sigma12 and at least one evolution parameter specified.
        (This is a case of redundancy which we, to be safe with consistency,
            do not allow).
    2. Both omega_i and Omega_i are given for at least one value of i.
        (Again, this is redundancy that we do not allow.)
    3. All three of the following are defined: h, density in DE, and density
        in curvature.
        (Again, redundancy. Technically, baryons and CDM also factor into h,
            but since those parameters are shape parameters, we set them to
            default values very early on in the strange case that they are
            missing.)
    """
    # We may want to allow the user to turn off error checking if a lot of
    # predictions need to be tested all at once...
    
    # Make sure that the user EITHER specifies sigma12 or ev. param.s
    if "sigma12" in cosmo_dict and contains_ev_par(cosmo_dict):
        raise ValueError(ambiguous_sigma12_msg)
    
    # Make sure that no parameters are doubly-defined
    if "omega_b" in cosmo_dict and "Omega_b" in cosmo_dict:
        raise ValueError(str.format(doubly_defined_msg, "baryons"))
    if "omega_cdm" in cosmo_dict and "Omega_cdm" in cosmo_dict:
        raise ValueError(str.format(doubly_defined_msg,
                                      "cold dark matter"))
    if "omega_DE" in cosmo_dict and "Omega_DE" in cosmo_dict:
        raise ValueError(str.format(doubly_defined_msg, "dark energy"))
    if "omega_K" in cosmo_dict and "Omega_K" in cosmo_dict:
        raise ValueError(str.format(doubly_defined_msg, "curvature"))
    if "h" in cosmo_dict and "H0" in cosmo_dict:
        raise ValueError("Do not specify h and H0 simultaneously. " + \
            "Specify one, and Cassandra-Linear will automatically handle " + \
            "the other.")

    # Make sure at most two of the three are defined: h, omega_curv, omega_DE
    if "h" in cosmo_dict or "H0" in cosmo_dict:
        if "Omega_DE" in cosmo_dict or "omega_DE" in cosmo_dict:
            if "Omega_K" in cosmo_dict or "omega_k" in cosmo_dict:
                raise ValueError(spec_conflict_msg)
    

fractional_keys = ["Omega_b", "Omega_cdm", "Omega_DE", "Omega_K", \
                   "Omega_nu"]
physical_keys = ["omega_b", "omega_cdm", "omega_DE", "omega_K", \
                 "omega_nu"]
   
def convert_densities(cosmo_dict):
    """
    Convert any fractional densities specified in cosmo_dict, and raise an
    error if there exists a fractional density without an accompanying h.
    """
    conversions = cp.deepcopy(cosmo_dict)
    
    # If h is present, set it right away, so that we can begin converting
    # fractional densities.
    if "H0" in conversions:
        conversions["h"] = conversions["H0"] / 100
 
    for i in range(len(fractional_keys)):
        frac_key = fractional_keys[i]
        if frac_key in conversions:
            # Make sure that h is present, in the event that a fractional
            # density parameter was given.
            if "h" not in conversions:
                raise ValueError(missing_h_message)
            phys_key = physical_keys[i]
            conversions[phys_key] = conversions[frac_key] * conversions["h"] ** 2
    
    return conversions
   
missing_h_message = "A fractional density parameter was specified, but no " + \
    "value of 'h' was provided."    
   
def fill_in_defaults(cosmo_dict):
    """
    Take an input cosmology and fill in missing values with defaults until it
    meets the requirements for emu prediction. 
    """
    conversions = cp.deepcopy(cosmo_dict)
    
    if "omega_b" not in conversions:
        warnings.warn(str.format(missing_shape_message, "omega_b"))
        conversions["omega_b"] = DEFAULT_COSMOLOGY["omega_b"]
    elif not within_prior(conversions["omega_b"], 0):
        raise ValueError(str.format(out_of_bounds_msg, "omega_b"))

    # Ditto with cold dark matter.
    if "omega_cdm" not in conversions:
        warnings.warn(str.format(missing_shape_message, "omega_cdm"))
        conversions["omega_cdm"] = DEFAULT_COSMOLOGY["omega_cdm"]
    elif not within_prior(conversions["omega_cdm"], 1):
        raise ValueError(str.format(out_of_bounds_msg, "omega_cdm"))

    # Ditto with the spectral index.
    if "ns" not in conversions:
        warnings.warn(str.format(missing_shape_message, "ns"))
        conversions["ns"] = DEFAULT_COSMOLOGY["ns"]
    elif not within_prior(conversions["ns"], 2):
        raise ValueError(str.format(out_of_bounds_msg, "ns"))
    
    # Ditto with neutrinos.
    if "omega_nu" not in conversions:
        warnings.warn("The value of 'omega_nu' was not provided. Assuming " + \
                      "massless neutrinos...")
        conversions["omega_nu"] = DEFAULT_COSMOLOGY["omega_nu"]
    else:
        if "As" not in conversions:
            warnings.warn("The value of 'As' was not provided, even " + \
                          "though massive neutrinos were requested. " + \
                          "Setting to the Planck best fit value...")
        if not within_prior(conversions["omega_nu"], 5):
            raise ValueError(str.format(out_of_bounds_msg, "omega_nu"))
            
    if "As" in conversions and not within_prior(conversions["As"], 4):
        raise ValueError(str.format(out_of_bounds_msg, "As"))

    return conversions

def cosmology_to_Pk(**kwargs):
    """
    This fn wraps the trainer object...
    
    It automatically returns a prediction and the estimated uncertainty on that
    prediction. I still don't really know why we re-invented the wheel, when
    the GPR object itself gives an uncertainty. But let's see...
    """

    error_check_cosmology(kwargs)
    
    cosmo_dict = convert_densities(kwargs)
    cosmo_dict = fill_in_defaults(cosmo_dict)
    
    cosmology = transcribe_cosmology(cosmo_dict)
    
    if "sigma12" not in cosmology.pars:
        add_sigma12(cosmology)
    else:
        if not within_prior(cosmology.pars["sigma12"], 3):
            raise ValueError(str.format(out_of_bounds_msg, "sigma12"))

    emu_vector = cosmology_to_emu_vec(cosmology)
    
    # Can I use an API that I can't necessarily "see"? i.e. does it work to
    # access trainer fn.s if cassL-dev isn't installed?
    
    if len(emu_vector) == 6: # massive neutrinos
        return nu_trainer.p_emu.predict(emu_vector), \
            nu_trainer.unc_emu.predict(emu_vector)
    elif len(emu_vector) == 4: # massless neutrinos
        return zm_trainer.p_emu.predict(emu_vector), \
            zm_trainer.unc_emu.predict(emu_vector)


def add_sigma12(cosmology):
    """
    @cosmology should be a fully filled-in Brenda Cosmology object.
    """
    base = np.array([
        cosmology.pars["omega_b"],
        cosmology.pars["omega_cdm"],
        cosmology.pars["ns"]
    ])
    
    base_normalized = sigma12_trainer.p_emu.convert_to_normalized_params(base)

    # First, emulate sigma12 as though the evolution parameters were all given
    # by the current best fit in the literature.
    sigma12_m0 = sigma12_trainer.p_emu.predict(base_normalized)
    
    cosmology.pars["sigma12"] = scale_sigma12(cosmology, sigma12_m0)
    
    if not within_prior(cosmology.pars["sigma12"], 3):
        raise ValueError("The given evolution parameters are invalid " + \
            "because they result in a sigma12 value outside our priors. " + \
            "Try a less extreme configuration.")
    
    return cosmology
    
def scale_sigma12(cosmology, sigma12_m0):
    """
    Ideally, the user wouldn't use this function, it would automatically be
    called under the hood in the event that the user attempts to specify
    evolution parameters in addition to the mandatory shape params.

    Preferentially uses a specified 'h' to define omega_DE while leaving
    omega_K as the default value.
    
    @cosmology: Brenda Cosmology object
        This contains all of the cosmological parameters we'll need in order to
        compute the linear growth factor. 
    
    @sigma12_m0: float
        The sigma12 value returned by the sigma12 emulator. This is what the
        sigma12 value would be if we overwrote Aletheia model 0 (i.e. the
        Planck best fit parameters) with this cosmology's values for omega_b,
        omega_cdm, and n_s.
    
    :return:
    
    @scaled_sigma12: float
        An estimate of the sigma12 given the cosmological parameters associated
        with @cosmology.
    """
    a = 1.0 / (1.0 + cosmology["z"])
    
    #! Should I be concerned about this a0 parameter?
    # After some cursory tests, I found that it has very little impact.
    LGF = cosmology.growth_factor(a, a0=1e-3, solver='odeint')
    growth_ratio = LGF / DEFAULT_COSMOLOGY["LGF"]
    
    # If the user specified no A_s value, the following factor automatically
    # disappears because, in this case, transcribe_cosmology sets
    # cosmology["As"] = DEFAULT_COSMOLOGY["As"]
    As_ratio = cosmology["As"] / DEFAULT_COSMOLOGY["As"]
    
    return DEFAULT_COSMOLOGY["sigma12"] * growth_ratio * np.sqrt(As_ratio)

def cosmology_to_emu_vec(cosmology):
    """
    Turn an input cosmology into an input vector that the emulator understands
    and from which it can predict a power spectrum.
    
    This needs to handle normalization, too.
    """
    base = np.array([
        cosmology.pars["omega_b"],
        cosmology.pars["omega_cdm"],
        cosmology.pars["ns"],
        cosmology.pars["sigma12"]
    ])
    
    if cosmology["omega_nu"] == 0:
        return zm_trainer.p_emu.convert_to_normalized_params(base)
    else:
        extension = np.array([
            cosmology.pars["As"],
            cosmology.pars["omnu"]
        ])
        full_vector = np.append(base, extension)
        return nu_trainer.p_emu.convert_to_normalized_params(full_vector)


def transcribe_cosmology(cosmo_dict):
    """
    Turn a set of arguments into a complete Cosmology object. Cosmology
    objects follow a particular format for compatibility with the fn.s in
    this script (and in Brendalib) that return various power spectra power.
    
    This fn. thoroughly error-checks the arguments to verify that they
    represent a consistent and complete cosmology. Mostly, completeness is not
    a problem for the user, as missing parameters are generally inferred from
    the default cosmology. However, for example, this fn will complain if the
    user attempts to specify fractional density parameters without specifying
    the value of the Hubble parameter.
    
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
    
    if "w" in cosmo_dict:
        conversions["w0"] = cosmo_dict["w"]
    if "w0" in cosmo_dict:
        conversions["w0"] = cosmo_dict["w0"]
    if "wa" in cosmo_dict:
        conversions["wa"] = cosmo_dict["wa"]

    # If h is present, set it right away, so that we can begin converting
    # fractional densities.
    if "h" in cosmo_dict:
        conversions["h"] = cosmo_dict["h"]
    elif "H0" in cosmo_dict:
        conversions["h"] = cosmo_dict["H0"] / 100
    
    # Nothing else requires such conversions, so add the remaining values
    # directly to the working dictionary.
    for key, value in cosmo_dict.items():
        if key not in fractional_keys:
            conversions[key] = value
        
    if "z" not in cosmo_dict:
        warnings.warn("No redshift given. Using z=0...")
        conversions["z"] = 0

    conversions["omega_m"] = conversions["omega_b"] + \
        conversions["omega_cdm"] + conversions["omega_nu"]

    # The question is, when omega_K is not specified, should we immediately set
    # it to default, or immediately set h to default and back-calculate
    # curvature?
    if "omega_DE" in conversions and "omega_K" not in conversions:
        if "h" not in conversions:
            # use default value for h
            conversions["omega_K"] = DEFAULT_COSMOLOGY["omega_K"]
        else:
            conversions["omega_K"] = conversions["h"] ** 2 - \
                conversions["omega_m"] - conversions["omega_DE"]

    # Analogous block for dark energy
    if "omega_K" in conversions and "omega_DE" not in conversions:
        if "h" not in conversions:
            # use default value for h
            conversions["h"] = DEFAULT_COSMOLOGY["h"]
        else:
            conversions["omega_DE"] = conversions["h"] ** 2 - \
                conversions["omega_m"] - conversions["omega_K"]

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
        DEFAULT_COSMOLOGY["h"] = np.sqrt(conversions["omega_m"] + \
            cosmology["omega_DE"] + cosmology["omega_K"])
        
    for i in range(len(physical_keys)):
        phys_key = physical_keys[i]
        frac_key = fractional_keys[i]
        if frac_key not in conversions:
            conversions[frac_key] = \
                conversions[phys_key] / conversions["h"] ** 2    
        
    # package it up for brenda
    if "As" not in conversions:
        conversions["As"] = DEFAULT_COSMOLOGY["As"]
    
    cosmology = brenda.Cosmology()
    conversions["Omega_m"] = conversions["omega_m"] / conversions["h"] ** 2
    conversions["de_model"] = "w0wa"
    # Brenda lib doesn't distinguish nu from CDM
    conversions["omega_cdm"] += conversions["omega_nu"]
    conversions["Omega_cdm"] += conversions["Omega_nu"]
    # The omegaK field will be ignored, but remembered through h
    # z will be ignored by brenda, but used by this code.
    cosmology.pars = conversions
        
    return cosmology

