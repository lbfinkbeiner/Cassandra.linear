import numpy as np
from cassL import lhc
from cassL import generate_emu_data as ged
from cassL import train_emu as te
from cassL import camb_interface as ci
import os

# One of the benefits of distancing ourselves from the camb naming scheme is
# it makes debugging easier: we'll quickly understand whether there is a problem
# with the dev code or the user code.
default_cosmology = {
    'omB': 0.022445,
    'omC': 0.120567,
    'n_s': 0.96,
    'A_s': 2.127238e-9,
    'omK': 0,
    'omDE': 0.305888,
    'h': 0.67
}

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
    raise NotImplementedError

    return OmM_0 * (1 + z) ** 3 + OmK_0 (1 + z) ** 2 + OmDE_0


def linear_growth_factor(OmM_0, z):
    raise NotImplementedError

    def integrand(zprime):
        return (1 + zprime) / E2(z) ** 1.5

    return 2.5 * OmM_0 * np.sqrt(E2(z)) * integrate(z, infinity, integrand)


spec_conflict_message = "Do not attempt to simultaneously set curvature, " + \
    "dark energy, and the Hubble parameter. Set two of the three, and " + \
    "Cassandra-Linear will automatically handle the third."


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
    cosmology = ci.default_cosmology()

    # Make sure at most two of the three are defined: h, omega_curv, omega_DE
    if "h" in kwargs:
        if "OmDE" or "omDE"in kwargs:
            if "OmK" or "omk" in kwargs:
                raise ValueError(spec_conflict_message)

    # Make sure that h is specified, in the event that fractionals are given
    fractional_keys = ["OmB", "OmC", "OmDE", "OmK"]
    fractional_in_kwargs = False

    for key in fractional_keys:
        if key in kwargs:
            fractional_in_kwargs = True
            break

    if fractional_in_kwargs and "h" not in kwargs:
        raise ValueError("A fractional density parameter was specified, " + \
            "but no value of 'h' was provided.")
            
    if "omB" in kwargs:
        cosmology["ombh2"] = kwargs["omB"]
    elif "OmB" in kwargs:
        cosmology["ombh2"] = kwargs["OmB"] / h ** 2
    else:
        raise ValueError("OmB is a required ingredient.")
    
    if "omC" in kwargs:
        cosmology["omch2"] = kwargs["omC"]
    elif "OmC" in kwargs:
        cosmology["omch2"] = kwargs["OmC"] / h ** 2
    else:
        raise ValueError("OmC is a required ingredient.")

    # Do likewise for DE and curvature, but instead of throwing an error, apply
    # the default value, i.e. that of Allie 0
    
    # Calculate h
    if "h" not in kwargs:
        cosmology["h"] = np.sqrt(omB + omC + omDE + omK)

    evolution_dictionary = {}

    # for key, item in kwargs:
        

    default_sigma12 = def_cosm[""]

    return old_sigma12 * linear_growth_factor(z) / linear_growth_factor(0)

    # We need to compute a ratio of the squares of the growth factors, right?
    
    # Evolution parameters to handle
    # w_a
    # w_0
    # h, i.e. Omega_DE
    # Omega_K
    
    # The fitting formula that Andres gives applies to a FLAT LambdaCDM
    # cosmology... what should I do?
    raise NotImplementedError

