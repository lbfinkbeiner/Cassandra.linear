import numpy as np
from cassL import lhc
from cassL import generate_emu_data as ged
from cassL import train_emu as te
from cassL import camb_interface as ci
import os

import scipy

# One of the benefits of distancing ourselves from the camb naming scheme is
# it makes debugging easier: we'll quickly understand whether there is a problem
# with the dev code or the user code.
DEFAULT_COSMOLOGY = {
    'omB': 0.022445,
    'omC': 0.120567,
    'n_s': 0.96,
    'A_s': 2.127238e-9,
    'omK': 0,
    # 'omDE': 0.305888,
    'h': 0.67
}

DEFAULT_SIGMA12 = 0.82476394

# linear growth factor
DEFAULT_LGF2 = 0.6238849955305038

data_prefix = os.path.dirname(os.path.abspath(__file__)) + "/"

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


def prior_file_to_dict(prior_name="COMET"):
    """
    Legacy function, use prior_file_to_array now instead.
    !
    Return a dictionary of arrays where each key is a cosmological parameter
    over which the emulator will be trained. The first entry of each array is
    the parameter's lower bound, the second entry is the parameter's upper
    bound.

    @priors string indicating which set of parameters to use.
        "MEGA": the original goal for this project, which unfortunately
            suffered from too many empty cells. The code has gone through
            several crucial bug fixes since switching to a different set of
            priors, so we need to test this prior suite again and re-assess the
            rate of empty cells.
        "classic": a prior range with widths in between those of "COMET" and
            "MEGA." We need to test this prior suite again to see if it still
            suffers from a large number of empty cells.
        "COMET" as of 19.06.23, this is the default for the emulator. It is
            the most restrictive of the three options and is intended to
            totally eliminate the problem of empty cells, so that a complete
            LHC can be used to train a demonstration emulator. The hope is for
            the demonstration emulator trained over such priors to be extremely
            accurate due to the very narrow permissible parameter values.

    @massive_neutrinos should be set to False when one is training the emulator
        for massless neutrinos. This is because the massless neutrino emulator
        should be using two fewer parameters--A_s and omega_nu_h2 are no longer
        appropriate.
    """
    param_ranges = {}

    prior_file = "priors/" + prior_name + ".txt"
    
    with open(prior_file, 'r') as file:
        lines = file.readlines()
        key = None
        for line in lines:
            if line[0] == "$":
                key = line[1:].strip()
            else:
                bounds = line.split(",")
                param_ranges[key] = [float(bounds[0]), float(bounds[1])]

    return param_ranges


def check_existing_files(scenario_name):    
    save_path = "data_sets/" + scenario_name
    
    train_complete = False
    test_complete = False
    
    if os.path.exists(save_path + "/lhc_train_final.npy") and \
        os.path.exists(save_path + "/samples_train.npy"):
        train_complete = True
        
    if os.path.exists(save_path + "/lhc_test_final.npy") and \
        os.path.exists(save_path + "/samples_test.npy"):
        test_complete = True
    
    return train_complete, test_complete
    

def get_scenario(scenario_name):
    scenario = {}
    file_handle = "scenarios/" + scenario_name + ".txt"
    
    with open(file_handle, 'r') as file:
        lines = file.readlines()
        key = None
        for line in lines:
            if line[0] == "#":
                continue
            if line.strip() == "":
                continue

            if line[0] == "$":
                key = line[1:].strip()
            else:
                val = line.strip()
                if val == "None":
                    val = None
                elif val.isnumeric() and key != "num_spectra_points":
                    val = float(val)
                scenario[key] = val
                
    return scenario
    

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
    
    data_dict["priors"] = prior_file_to_dict(scenario["priors"])

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
    "fractional density in {}. Specify one or the other, and " + \
    "Cassandra-Linear will automatically handle the other."
    
missing_h_message = "A fractional density parameter was specified, but no " + \
    "value of 'h' was provided."
    
missing_shape_message = "The value of {} was not provided. This is an " + \
    "emulated shape parameter and is required. Setting to the Planck " + \
    "best-fit value."

def scale_sigma12(**kwargs):
    """
    Ideally, the user wouldn't use this function, it would automatically be
    called under the hood in the event that the user attempts to specify
    evolution parameters in addition to the mandatory shape params.

    Preferentially uses a specified 'h' to define omega_DE while leaving omega_K
    as the default value.
    :param kwargs:
    :return:
    raise Warning("Ouch!")
    print("Hello")
    """
    # This is an arbitrarily-formatted dictionary just for internal use in this
    # function; it helps to keep track of values that may need to be converted
    # or inferred from others.
    conversions = {}

    # Make sure that no density parameters are doubly-defined
    if "omB" in kwargs and "OmB" in kwargs:
        raise ValueError(str.format(doubly_defined_message, "baryons"))
    if "omC" in kwargs and "OmC" in kwargs:
        raise ValueError(str.format(doubly_defined_message, "cold dark matter"))
    if "omDE" in kwargs and "OmDE" in kwargs:
        raise ValueError(str.format(doubly_defined_message, "dark energy"))
    if "omK" in kwargs and "OmK" in kwargs:
        raise ValueError(str.format(doubly_defined_message, "curvature"))

    # Make sure at most two of the three are defined: h, omega_curv, omega_DE
    if "h" in kwargs:
        if "OmDE" or "omDE"in kwargs:
            if "OmK" or "omk" in kwargs:
                raise ValueError(spec_conflict_message)

    # If h is present, set it right away, so that we can begin converting
    # fractional densities.
    if "h" in kwargs:
        conversions["h"] = kwargs["h"]

    # Make sure that h is present, in the event that fractionals are given
    fractional_keys = ["OmB", "OmC", "OmDE", "OmK"]
    physical_keys = ["omB", "omC", "omDE", "omK"]
    
    for i in range(len(fractional_keys)):
        frac_key = fractional_keys[i]
        if key in kwargs:
            if "h" not in conversions:
                raise ValueError(missing_h_message)
            phys_key = physical_keys[i]
            conversions[phys_key] = kwargs[frac_key] / conversions["h"] ** 2
    
    # Nothing else requires such conversions, so add the remaining values
    # directly to the working dictionary.
    
    for key, value in kwargs.items():
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
        warnings.warn(str.format(mising_shape_message, "n_s"))
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
                np.sqrt(conversions["h"] ** 2 - omM - conversions["omDE"])

    # Analogous block for dark energy
    if "omK" in conversions and "omDE" not in conversions:
        if "h" not in conversions:
            # use default value for h
            conversions["h"] = DEFAULT_COSMOLOGY["h"]
        else:
            conversions["omDE"] = \
                np.sqrt(conversions["h"] ** 2 - omM - conversions["omK"]

    # Fill in default values for density parameters, because we need these to
    # compute h
    if "omK" not in conversions:
        conversions["omK"] = DEFAULT_COSMOLOGY["omK"]

    # If omDE was never given, there's no point in calculating h
    if "omDE" not in conversions:
        conversions["h"] = DEFAULT_COSMOLOGY["h"]

    # If h wasn't given, compute it now that we have all of the physical
    # densities.
    if "h" not in conversions:
        DEFAULT_COSMOLOGY["h"] = np.sqrt(omB + omC + omDE + omK)

    OmM = omM / conversions["h"] ** 2
    OmK = conversions["omK"] / conversions["h"] ** 2
    OmDE = conversions["omDE"] / conversions["h"] ** 2
    LGF2 = linear_growth_factor(OmM, OmK, OmDE, conversions["z"]) ** 2
    return DEFAULT_SIGMA12 * LGF2 / DEFAULT_LGF2

    # Evolution parameters to handle
    # w_a
    # w_0
    raise NotImplementedError

