import numpy as np
from cassL import lhc
from cassL import generate_emu_data as ged
from cassL import train_emu as te
from cassL import camb_interface as ci
import os

#!! Matteo's code, which still needs to be gracefully incorporated
import cosmo_tools

import scipy
import warnings

# One of the benefits of distancing ourselves from the camb naming scheme is
# it makes debugging easier: we'll quickly understand whether there is a problem
# with the dev code or the user code.
DEFAULT_COSMOLOGY = {
    'omB': 0.022445,
    'omC': 0.120567,
    'n_s': 0.96,
    'A_s': 2.127238e-9,
    'omK': 0,
    'omDE': 0.305888,
    'h': 0.67
}

DEFAULT_SIGMA12 = 0.82476394
# linear growth factor
DEFAULT_LGF = 0.7898639094999238

data_prefix = os.path.dirname(os.path.abspath(__file__)) + "/"

# Evolution parameter keys. If ANY of these appears in a cosmology dictionary,
# there had better not be a sigma12 value...
EV_PAR_KEYS = []

def contains_ev_par(dictionary):
    for ev_par_key in EV_PAR_KEYS:
        if ev_par_key in dictionary:
            return true

def dictionary_to_emu_vec(dictionary):
    # Regardless of whether we use the massless or massive emu,
    # we need omega_b, omega_c, n_s, and sigma12
    
    # We may want to allow the user to turn off error checking if a lot of
    # predictions need to be tested all at once...
    if "sigma12" in dictionary and contains_ev_par(dictionary)

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
    
    # Wait a moment... isn't the saved emu file supposed to contain
    # both the primary and unc emu's??
    # Maybe we simply did this on a different machine...
    nu_trainer = np.load(data_prefix + "emus/Hnu2.cle")
    z_trainer = np.load(data_prefix + "emus/Hz1.cle")
    
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
    Turn a set of arguments into a complete cosmology dictionary. Cosmology
    dictionaries follow a particular format for compatibility with the fn.s in
    this script that return various power spectra power.
    
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
    
    omK: float
        Physical density in curvature
    OmK: float
        Fractional density in curvature
    
    h: float
        dimensionless Hubble parameter
    H0: float
        Hubble parameter in km / s / Mpc
        
    n_s: float
        Spectral index of the primordial power spectrum
    
    A_s: float
        Scalar mode amplitude of the primordial power spectrum
        
    !!!
    z: float
        redshift. THIS PROBABLY DOESN'T BELONG IN THE COSMOLOGY DICTIONARY.
        Maybe we should leave redshift as a separate input in the various fn.s
        of this script?
    """
    # To-do: add support for "wa" and "w0"
    if "w" in kwargs or "w0" in kwargs or "wa" in kwargs:
        raise NotImplementedError("This fn does not yet support DE EoS " + \
                                    "customization.")

    # This is an arbitrarily-formatted dictionary just for internal use in this
    # function; it helps to keep track of values that may need to be converted
    # or inferred from others.
    conversions = {}

    # Make sure that no parameters are doubly-defined
    if "omB" in kwargs and "OmB" in kwargs:
        raise ValueError(str.format(doubly_defined_message, "baryons"))
    if "omC" in kwargs and "OmC" in kwargs:
        raise ValueError(str.format(doubly_defined_message, "cold dark matter"))
    if "omDE" in kwargs and "OmDE" in kwargs:
        raise ValueError(str.format(doubly_defined_message, "dark energy"))
    if "omK" in kwargs and "OmK" in kwargs:
        raise ValueError(str.format(doubly_defined_message, "curvature"))
    if "h" in kwargs and "H0" in kwargs:
        raise ValueError("Do not specify h and H0 simultaneously. Specify " + \
            "one, and Cassandra-Linear will automatically handle the other.")

    # Make sure at most two of the three are defined: h, omega_curv, omega_DE
    if "h" in kwargs or "H0" in kwargs:
        if "OmDE" in kwargs or "omDE" in kwargs:
            if "OmK" in kwargs or "omk" in kwargs:
                raise ValueError(spec_conflict_message)

    # If h is present, set it right away, so that we can begin converting
    # fractional densities.
    if "h" in kwargs:
        conversions["h"] = kwargs["h"]
    elif "H0" in kwargs:
        conversions["h"] = kwargs["H0"] / 100

    # Make sure that h is present, in the event that fractionals are given
    fractional_keys = ["OmB", "OmC", "OmDE", "OmK"]
    physical_keys = ["omB", "omC", "omDE", "omK"]
    
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
    
    # Now complain about missing entries.
    # We have to fill in missing densities immediately because they will be
    # used to find h.
    if "omB" not in kwargs:
        warnings.warn(str.format(missing_shape_message, "omB"))
        conversions["omB"] = DEFAULT_COSMOLOGY["omB"]

    # Ditto with cold dark matter.
    if "omC" not in kwargs:
        warnings.warn(str.format(missing_shape_message, "omC"))
        conversions["omC"] = DEFAULT_COSMOLOGY["omC"]

    # Ditto with the spectral index.
    if "n_s" not in kwargs:
        warnings.warn(str.format(missing_shape_message, "n_s"))
        conversions["n_s"] = DEFAULT_COSMOLOGY["n_s"]

    if "z" not in kwargs:
        warnings.warn("No redshift given. Using z=0...")
        conversions["z"] = 0

    omM = conversions["omB"] + conversions["omC"]

    # The question is, when omK is not specified, should we immediately set it
    # to default, or immediately set h to default and back-calculate curvature?
    if "omDE" in conversions and "omK" not in conversions:
        if "h" not in conversions:
            # use default value for h
            conversions["omK"] = DEFAULT_COSMOLOGY["omK"]
        else:
            conversions["omK"] = \
                conversions["h"] ** 2 - omM - conversions["omDE"]

    # Analogous block for dark energy
    if "omK" in conversions and "omDE" not in conversions:
        if "h" not in conversions:
            # use default value for h
            conversions["h"] = DEFAULT_COSMOLOGY["h"]
        else:
            conversions["omDE"] = \
                conversions["h"] ** 2 - omM - conversions["omK"]

    # Fill in default values for density parameters, because we need these to
    # compute h
    if "omK" not in conversions:
        conversions["omK"] = DEFAULT_COSMOLOGY["omK"]

    # If omDE was never given, there's no point in calculating h
    if "omDE" not in conversions:
        conversions["h"] = DEFAULT_COSMOLOGY["h"]
        conversions["omDE"] = DEFAULT_COSMOLOGY["omDE"]

    # If h wasn't given, compute it now that we have all of the physical
    # densities.
    if "h" not in conversions:
        DEFAULT_COSMOLOGY["h"] = np.sqrt(omB + omC + omDE + omK)
        
    return conversions

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

    omM = conversions["omB"] + conversions["omC"]
    OmM = omM / conversions["h"] ** 2
    OmK = conversions["omK"] / conversions["h"] ** 2
    OmDE = conversions["omDE"] / conversions["h"] ** 2
    
    LGF = linear_growth_factor(OmM, OmK, OmDE, conversions["z"])
    As_ratio = conversions["A_s"] / DEFAULT_COSMOLOGY["A_s"]
    
    return DEFAULT_SIGMA12 * LGF / DEFAULT_LGF * np.sqrt(As_ratio)

