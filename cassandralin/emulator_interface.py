import numpy as np

import os
import copy as cp
import warnings

#!! Matteo's code, which still needs to be gracefully incorporated
import cosmo_tools as brenda

DATA_PREFIX = os.path.dirname(os.path.abspath(__file__)) + "/"

# Load the array of inverse scale values at which the emulator makes P(k)
# predictions. This array should contain NumPy floats, so there's nothing to
# unpickle.
K_AXIS = np.load(DATA_PREFIX + "300k.npy")

# Load the emulators that we need. The extension "trainer" refers to the
# data structure which encapsulates the emu. The user may refer to
# "train_emu.py" from the developer tools for implementation details, but these
# details are not necessary for the usage of this script.
# sigma12 emu
SIGMA12_TRAINER = np.load(DATA_PREFIX + "emus/sigma12.cle", allow_pickle=True)
# Massive-neutrino emu
NU_TRAINER = np.load(DATA_PREFIX + "emus/Hnu2.cle", allow_pickle=True)
# Zero-mass neutrino emu
ZM_TRAINER = np.load(DATA_PREFIX + "emus/Hz1.cle", allow_pickle=True)

# Aletehia model 0 parameters, given by the best fit to the Planck data.
DEFAULT_COSMO_DICT = {
    'omega_b': 0.022445,
    'omega_cdm': 0.120567,
    'ns': 0.96,
    'As': 2.127238e-9,
    'omega_K': 0.,
    'omega_DE': 0.305888,
    'omega_nu': 0.,
    'h': 0.67,
    'z': 0.,
    'sigma12': 0.82476394,
}

DEFAULT_BRENDA_COSMO = transcribe_cosmology(DEFAULT_COSMO_DICT)    

# Evolution parameter keys. If ANY of these appears in a cosmology dictionary,
# the dictionary had better not include a sigma12 value...
#! Should h be here? It's only an evolution parameter because of our particular
# implementation of evolution mapping...
EV_PAR_KEYS = ["omega_K", "Omega_K", "omega_DE", "Omega_DE", "w", "w0", 
    "wa", "h", "H0"]

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

    :param prior_name: The name of the prior file to be read, i.e. the file
        handle minus the file extension, defaults to "COMET".
    :type prior_name: str, optional
    :return: Priors associated with the given @prior_name. The first index
        determines the cosmological parameter and the second index is either 0
        for the lower bound or 1 for the upper bound.
    :rtype: numpy.ndarray of float64
    """
    param_ranges = None

    prior_file = DATA_PREFIX + "priors/" + prior_name + ".txt"
    
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
    
    :param value: The parameter value to be tested
    :type value: float
    :param index: The index of PRIORS (the array of priors over which the
        emulator was trained) which corresponds to that cosmological parameter.
    :type index: int
    :return: Wether @value is within the associated prior range for the
        parameter specified by @index.
    :rtype: bool

    For example, if @value were a configuration of the spectral index. We would
        call within_prior(value, 2).
    """
    return value >= PRIORS[index][0] and value <= PRIORS[index][1]

def neutrinos_massive(cosmo_dict):
    """
    Ascertain whether the provided cosmology dictionary corresponds to a
    cosmology with neutrinos of nonzero mass.
    
    :param cosmo_dict: Dictionary giving values of cosmological parameters,
        where the parameters are referred to using the same keys as Brendalib
        does in its Cosmology objects.
    :type cosmo_dict: dict
    :return: Whether cosmo_dict has massive neutrinos.
    :rtype: bool
    """
    if "omega_nu" in cosmo_dict:
        return cosmo_dict["omega_nu"] > 0
    elif "Omega_nu" in cosmo_dict:
        return cosmo_dict["Omega_nu"] > 0
    # We could find neither a fractional nor physical density. What is the
    # default value?
    return DEFAULT_COSMO_DICT["omega_nu"] > 0


def contains_ev_par(cosmo_dict):
    """
    Check if the cosmology specified by @cosmo_dict contains a definition of
    an evolution parameter.

    :param cosmo_dict: Dictionary giving values of cosmological parameters,
        where the parameters are referred to using the same keys as Brendalib
        does in its Cosmology objects.
    :type cosmo_dict: dict
    :return: Whether @cosmo_dict contains a key for an evolution parameter.
    :rtype: bool
    """
    for ev_par_key in EV_PAR_KEYS:
        if ev_par_key in cosmo_dict:
            return 1
       
    if "z" in cosmo_dict:
        return 2
       
    # This is a special case: when the neutrinos are massless, As behaves as an
    # evolution parameter.
    if "As" not in cosmo_dict and neutrinos_massive(cosmo_dict):
        return 3
            
    return 0

doubly_defined_msg = "Do not simultaneously specify the physical and " + \
    "fractional density in {}. Specify one, and " + \
    "Cassandra-Linear will automatically handle the other."
    
out_of_bounds_msg = "The given value for {} falls outside of the range " + \
    "over which the emulators were trained. Try a less extreme value."

def error_check_cosmology(cosmo_dict):
    """
    Provide clear error messages for some, but not all, cases of ambiguity or
    inconsistency in the input parameters:
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
    These error messages are given via raises, so this fn is void. The
    input @cosmo_dict is considered to be valid if this fn returns None.

    Mostly, completeness is not a problem for which we need error checking,
    as missing parameters are generally inferred from the default (Planck best
    fit) cosmology. As an exception, we do not infer h when the user specifies
    fractional density parameters (see convert_fractional_densities).

    :param cosmo_dict: dictionary giving values of cosmological parameters,
        where the parameters are referred to using the same keys as Brendalib
        does in its Cosmology objects.
    :type cosmo_dict: dict
    :raises: ValueError
        A ValueError is raised in all of the three cases specified at the
        beginning of this docstring.
    """
    # We may want to allow the user to turn off error checking if a lot of
    # predictions need to be tested all at once.... However, if we add such a
    # feature, we'll need to bring more of the error checking into this fn,
    # because currently some checks are scattered in other functions...
    # 1. transcribe_cosmology warns in the absence of As
    # 2. convert_fractional_densities raises missing_h error
    # 3. fill_in_defaults raises out_of_bounds errors and missing shape
    #   warnings
    # 4. cosmology_to_Pk raises out_of_bounds error for sigma12
    # 5. add_sigma12 raises an error if the evolution parameters are too
    #   extreme.
    # Probably, you won't be able to collect all of those errors here, but you
    # will probably be able to extract most of them.
    
    # Make sure that the user EITHER specifies sigma12 or ev. param.s
    evolution_code = contains_ev_par(cosmo_dict)
    if "sigma12" in cosmo_dict and evolution_code(cosmo_dict):
        if evolution_code == 1:
            raise ValueError("The sigma12 and at least one evolution " + \
                "parameter were simultaneously specified. If the desired " + \
                "sigma12 is already known, no evolution parameters should " + \
                "appear here.")
        elif evolution_code == 2:
            raise ValueError("The sigma12 and redshift were " + \
                "simultaneously specified. Only one of the two should be " + \
                "given, because redshift behaves as an evolution parameter.")
        elif evolution_code == 3:
            raise ValueError("The sigma12 and As were simulataneously " + \
                "specified, but the neutrinos are massless in this " + \
                "cosmology. In this case, As behaves as an evolution " + \
                "parameter, so only one of the two can be specified.")
    
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
                raise ValueError("Do not attempt to simultaneously set " + \
                    "curvature, dark energy, and the Hubble parameter. " + \
                    "Set two of the three, and Cassandra-Linear will " + \
                    "automatically handle the third.")
    

fractional_keys = ["Omega_b", "Omega_cdm", "Omega_DE", "Omega_K", \
                   "Omega_nu"]
physical_keys = ["omega_b", "omega_cdm", "omega_DE", "omega_K", \
                 "omega_nu"]

missing_h_message = "A fractional density parameter was specified, but no " + \
    "value of 'h' was provided."    

def convert_fractional_densities(cosmo_dict):
    """
    Convert any fractional densities specified in cosmo_dict, and raise an
    error if there exists a fractional density without an accompanying h.
    
    :param cosmo_dict: dictionary giving values of cosmological parameters,
        where the parameters are referred to using the same keys as Brendalib
        does in its Cosmology objects.
    :type cosmo_dict: dict
    :raises: ValueError
        This error is raised if there exists a fractional density without an
        accompanying value of h.
    :return: copy of @cosmo_dict with additional fields for the computed
        physical densities. If @cosmo_dict already contains all physical
        densities, or if no fractional densities were specified, this returned
        copy will be indistinguishable from @cosmo_dict.
    :rtype: dictionary
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
            conversions[phys_key] = \
                conversions[frac_key] * conversions["h"] ** 2
    
    return conversions

missing_shape_message = "The value of {} was not provided. This is an " + \
    "emulated shape parameter and is required. Setting to the Planck " + \
    "best-fit value ({})..."
   
def fill_in_defaults(cosmo_dict):
    """
    Take an input cosmology and fill in missing values with defaults until it
    meets the requirements for emu prediction. The parameters considered
    required (and therefore subject to filling in with default values) depend
    on the neutrino treatment.

    If this cosmology uses massless neutrinos, the following parameters are
    required: omega_b, omega_cdm, and ns (technically, sigma12 is also
    required, but instead of filling that in with a default value, we emulate
    it and analytically rescale the prediction with the fn add_sigma12). If
    this cosmology uses massive neutrinos, two parameters are required in
    addition: As and omega_nu.

    The default values for all of these required parameters are recorded in the
    DEFAULT_COSMO_DICT dictionary at the top of this script. For example, if
    @cosmo_dict lacks an entry for omega_nu, neutrinos are assumed to be
    massless and omega_nu is set to zero. 
    
    :param cosmo_dict: dictionary giving values of cosmological parameters,
        where the parameters are referred to using the same keys as Brendalib
        does in its Cosmology objects.
    :type cosmo_dict: dict
    :raises: ValueError
        This error is raised if any of the required parameters fall outside
        the range defined by the priors over which the emulators were trained.
    :return: copy of @cosmo_dict with additional fields for the default
        parameters. If @cosmo_dict already contains all required parameters,
        this returned copy will be indistinguishable from @cosmo_dict.
    :rtype: dictionary
    warning:: This fn will fill-in with defaults the values of essential shape
        parameters like omega_b and omega_cdm which are directly used by the
        emulator, unlike the less consequential evolution parameters which
        only affect the predicted spectra by an analytical shift in amplitude.
        This fn will warn the user if it is filling in these parameters with
        default values, because that almost certainly indicates that the user
        has forgotten to provide them.
    """
    conversions = cp.deepcopy(cosmo_dict)
    
    if "omega_b" not in conversions:
        warnings.warn(str.format(missing_shape_message, "omega_b",
            DEFAULT_COSMO_DICT["omega_b"]))
        conversions["omega_b"] = DEFAULT_COSMO_DICT["omega_b"]
    elif not within_prior(conversions["omega_b"], 0):
        raise ValueError(str.format(out_of_bounds_msg, "omega_b"))

    # Ditto with cold dark matter.
    if "omega_cdm" not in conversions:
        warnings.warn(str.format(missing_shape_message, "omega_cdm",
            DEFAULT_COSMO_DICT["omega_cdm"]))
        conversions["omega_cdm"] = DEFAULT_COSMO_DICT["omega_cdm"]
    elif not within_prior(conversions["omega_cdm"], 1):
        raise ValueError(str.format(out_of_bounds_msg, "omega_cdm"))

    # Ditto with the spectral index.
    if "ns" not in conversions:
        warnings.warn(str.format(missing_shape_message, "ns",
            DEFAULT_COSMO_DICT["ns"]))
        conversions["ns"] = DEFAULT_COSMO_DICT["ns"]
    elif not within_prior(conversions["ns"], 2):
        raise ValueError(str.format(out_of_bounds_msg, "ns"))
    
    # Ditto with neutrinos.
    if "omega_nu" not in conversions:
        warnings.warn("The value of 'omega_nu' was not provided. Assuming " + \
                      "a value of " + str(DEFAULT_COSMO_DICT["omega_nu"]) + \
                      "...")
        conversions["omega_nu"] = DEFAULT_COSMO_DICT["omega_nu"]
    elif not within_prior(conversions["omega_nu"], 5):
        raise ValueError(str.format(out_of_bounds_msg, "omega_nu"))

    if "As" not in conversions and neutrinos_massive(conversions):
        warnings.warn("The value of 'As' was not provided, even " + \
                      "though massive neutrinos were requested. " + \
                      "Setting to the Planck best fit value (" + \
                      str(DEFAULT_COSMO_DICT["As"]) + ")...")
        conversions["As"] = DEFAULT_COSMO_DICT["As"]
    elif "As" in conversions and not within_prior(conversions["As"], 4):
        raise ValueError(str.format(out_of_bounds_msg, "As"))

    return conversions


def cosmology_to_Pk(**kwargs):
    """
    Predict the power spectrum based on cosmological parameters provided by the
    user. The returned power spectrum is evaluated at 300 values of the inverse
    scale k, given by K_AXIS.
    
    Also return the estimated uncertainty on that prediction. This
    uncertainty is not the Gaussian Process Regression uncertainty, but an
    empirical uncertainty based on the emulator's performance on a set of
    validation data.
    
    Any cosmological parameters not specified by the user will be assigned
    default values according to Aletheia model 0, a cosmology based on the
    best fit to the Planck data but without massive neutrinos.
    
    :param omega_b: Physical density in baryons, defaults to
        DEFAULT_COSMO_DICT['omega_b']
    :type omega_b: float
    :param Omega_b: Fractional density in baryons, defaults to
        DEFAULT_COSMO_DICT['omega_b'] / DEFAULT_COSMO_DICT['h'] ** 2
    :type Omega_b: float
    :param omega_cdm: Physical density in cold dark matter, defaults to
        DEFAULT_COSMO_DICT['omega_cdm']
    :type omega_cdm: float
    :param Omega_cdm: Fractional density in cold dark matter, defaults to
        DEFAULT_COSMO_DICT['omega_cdm'] / DEFAULT_COSMO_DICT['h'] ** 2
    :type Omega_cdm: float
    :param omega_DE: Physical density in dark energy, defaults to
        DEFAULT_COSMO_DICT['omega_DE']
    :type omega_DE: float
    :param Omega_DE: Fractional density in dark energy, defaults to
        DEFAULT_COSMO_DICT['omega_DE'] / DEFAULT_COSMO_DICT['h'] ** 2
    :type Omega_DE: float
    :param omega_nu: Physical density in neutrinos, defaults to
        DEFAULT_COSMO_DICT['omega_nu']
    :type omega_nu: float
    :param Omega_nu: Fractional density in neutrinos, defaults to
        DEFAULT_COSMO_DICT['omega_nu'] / DEFAULT_COSMO_DICT['h'] ** 2
    :type Omega_nu: float
    :param omega_K: Physical density in curvature, defaults to
        DEFAULT_COSMO_DICT['omega_K']
    :type omega_K: float
    :param Omega_K: Fractional density in curvature, defaults to
        DEFAULT_COSMO_DICT['omega_K'] / DEFAULT_COSMO_DICT['h'] ** 2
    :type Omega_K: float
    :param h: Dimensionless Hubble parameter, defaults to
        DEFAULT_COSMO_DICT['h']
    :type h: float
    :param H0: Hubble parameter in km / s / Mpc, defaults to
        DEFAULT_COSMO_DICT['h'] * 100
    :type H0: float
    :param ns: Spectral index of the primordial power spectrum, defaults to
        DEFAULT_COSMO_DICT['ns']
    :type ns: float
    :param As: Scalar mode amplitude of the primordial power spectrum, defaults
        to DEFAULT_COSMO_DICT['As']
    :type As: float
    :param z: Cosmological redshift, defaults to DEFAULT_COSMO_DICT['z']
    :type z: float
    :raises: ValueError
        This error is raised if the user explicitly specifies a value of
        sigma12 that falls outside of the prior range.
    :return: An array with the same length as K_AXIS, containing the
        values of P(k) emulated by Cassandra-Linear for each value of k in
        K_AXIS.
    :rtype: numpy.ndarray of float64
    
    todo:: Consider putting redshift somewhere else. It's conceptually unclean
        to make a single redshift value a part of the definition of the
        cosmology.
        Consider also including the GPy uncertainty in the output. Would that
        be helpful? Would that be more useful than the emulated uncertainty?
    """
    # If you need to speed up the predictions, it would be worthwhile to
    # consider the theoretically optimal case: In this case, the user would
    # have already error-checked and neatly packaged their data. So, to time
    # the theoretically optimal case is to time JUST the call
    # NU_TRAINER.delta_emu.predict(emu_vector)[0]

    error_check_cosmology(kwargs)
    
    cosmo_dict = convert_fractional_densities(kwargs)
    cosmo_dict = fill_in_defaults(cosmo_dict)
    
    cosmology = transcribe_cosmology(cosmo_dict)
    
    if "sigma12" not in cosmology.pars:
        cosmology = add_sigma12(cosmology)
    else:
        if not within_prior(cosmology.pars["sigma12"], 3):
            raise ValueError(str.format(out_of_bounds_msg, "sigma12"))

    emu_vector = cosmology_to_emu_vec(cosmology)
    
    # Again, we have to de-nest
    if len(emu_vector) == 6: # massive neutrinos
        return NU_TRAINER.p_emu.predict(emu_vector)[0], \
            NU_TRAINER.delta_emu.predict(emu_vector)[0]
    elif len(emu_vector) == 4: # massless neutrinos
        return ZM_TRAINER.p_emu.predict(emu_vector)[0], \
            ZM_TRAINER.delta_emu.predict(emu_vector)[0]


def add_sigma12(cosmology):
    """
    Estimate the sigma12 value given the parameters specified by @cosmology.
    This involves:
    1. An emulator prediction for sigma12 based on omega_b, omega_cdm, and n_s,
        where all other parameters are taken from the Planck best fit.
    2. An analytical rescaling of this prediction based on the provided
        evolution parameters (although, as part of this step, the values of
        omega_b and omega_cdm are used again).
    
    :param cosmology: A fully filled-in Brenda Cosmology object. It is not
        recommended to manually create this object, but to start with a
        cosmology dictionary (of the format used by DEFAULT_COSMO_DICT) and
        then to run it through the conversion functions
        convert_fractional_densities, fill_in_defaults, and
        transcribe_cosmology, optionally verifying the validity of @cosmology
        with error_check_cosmology. These functions facilitate the creation of
        a fully-specified Brenda Cosmology object.
    :type cosmology: instance of the Cosmology class from Brenda.
    :raises: ValueError
        This error is raised if the given evolution parameters are so extreme
        that the analytically-rescaled sigma12 value falls outside of the prior
        range.
    :return: A copy of @cosmology where @cosmology.pars contains a new field,
        "sigma12", an estimate of the sigma12 value associated with the
        cosmology.
    :rtype: instance of the Cosmology class from Brenda.
    """
    new_cosmology = cp.deepcopy(cosmology)
    
    base = np.array([
        new_cosmology.pars["omega_b"],
        new_cosmology.pars["omega_cdm"],
        new_cosmology.pars["ns"]
    ])
    
    base_normalized = SIGMA12_TRAINER.p_emu.convert_to_normalized_params(base)

    # First, emulate sigma12 as though the evolution parameters were all given
    # by the current best fit in the literature.
    
    # Extreme de-nesting required due to the format of the emu's.
    unscaled_sigma12 = SIGMA12_TRAINER.p_emu.predict(base_normalized)[0][0]
    
    new_cosmology.pars["sigma12"] = scale_sigma12(new_cosmology,
        unscaled_sigma12)
    
    if not within_prior(new_cosmology.pars["sigma12"], 3):
        raise ValueError("The given evolution parameters are invalid " + \
            "because they result in a sigma12 value outside our priors. " + \
            "Try a less extreme configuration.")
    
    return new_cosmology
    
    
def scale_sigma12(cosmology, old_sigma12):
    """
    Analytically solve for a new sigma12 value given a baseline (an emulated
    sigma12 value) and a new set of cosmological parameters.
    
    Ideally, the user wouldn't call this function explicitly. It would
    automatically be called under the hood by cosmology_to_Pk in the event that
    the user specifies evolution parameters instead of explicitly giving
    sigma12.
    
    :param cosmology: A fully filled-in Brenda Cosmology object. It is not
        recommended to manually create this object, but to start with a
        cosmology dictionary (of the format used by DEFAULT_COSMO_DICT) and
        then to run it through the conversion functions
        convert_fractional_densities, fill_in_defaults, and
        transcribe_cosmology, optionally verifying the validity of @cosmology
        with error_check_cosmology. These functions facilitate the creation of
        a fully-specified Brenda Cosmology object.
    :type cosmology: instance of the Cosmology class from Brenda.

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
    # In order to scale the sigma12 value, we'll need to calculate the LGF of
    # the emulated cosmology: this is DEFAULT_BRENDA_COSMO, but with the three
    # shape parameters specified (ns makes no difference but we include it here
    # for completeness).
    emu_cosmology = cp.deepcopy(DEFAULT_BRENDA_COSMO)
    for key in ["omega_b", "omega_cdm", "ns"]:
        emu_cosmology.pars[key] = cosmology.pars[key]
    
    new_a = 1.0 / (1.0 + cosmology.pars["z"])
    old_a = 1.0 / (1.0 + emu_cosmology.pars["z"])
    
    #! Should I be concerned about this a0 parameter?
    # After some cursory tests, I found that it has very little impact.
    # De-nest
    new_LGF = cosmology.growth_factor(new_a, a0=1e-3, solver='odeint')[0]
    old_LGF = emu_cosmology.growth_factor(old_a, a0=1e-3,
        solver='odeint')[0]
    growth_ratio = new_LGF / old_LGF
    
    # If the user specified no A_s value, the following factor automatically
    # disappears because, in this case, transcribe_cosmology sets
    # cosmology["As"] = DEFAULT_COSMO_DICT["As"]
    As_ratio = cosmology.pars["As"] / emu_cosmology["As"]
    
    return sigma12_m0 * growth_ratio * np.sqrt(As_ratio)


def cosmology_to_emu_vec(cosmology):
    """
    Turn an input cosmology into an input vector that the emulator understands
    and from which it can predict a power spectrum. This includes
    normalization, which is handled by the emulators themselves according to
    the priors that they've stored.
    
    @cosmology: a Brenda Cosmology object which should already contain the
        desired shape and evolution parameters.
    """
    base = np.array([
        cosmology.pars["omega_b"],
        cosmology.pars["omega_cdm"],
        cosmology.pars["ns"],
        cosmology.pars["sigma12"]
    ])
    
    if cosmology.pars["omega_nu"] == 0:
        return ZM_TRAINER.p_emu.convert_to_normalized_params(base)
    else:
        extension = np.array([
            cosmology.pars["As"],
            cosmology.pars["omega_nu"]
        ])
        full_vector = np.append(base, extension)
        return NU_TRAINER.p_emu.convert_to_normalized_params(full_vector)


def transcribe_cosmology(cosmo_dict):
    """
    Turn a set of arguments into a Brenda Cosmology object. Cosmology
    objects follow a particular format for compatibility with the fn.s in
    this script (and in Brendalib) that return various power spectra power.
    
    After this verification, the fn converts given quantities to desired
    quantities. For example, the code in this script primarily uses physical 
    densities, so fractional densities will be converted.
    
    In the event that two of {"omega_K", "omega_DE" and "h"} are missing, 
    default values are filled in until the rest of the missing parameters can
    be inferred. Defaults are applied in this order:
    1. omega_K
    2. h
    3. omega_DE
    
    ### cosmo_dict should already have everything in place. This function does
        NOT fill in default values for essential parameters like omega_b
        (and indeed will raise an error if these are not provided),
        although it does fill in defaults in a oouple of evolution-parameter
        cases, such as omega_K.
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
        warnings.warn("No redshift given. Using " + \
            str(DEFAULT_COSMO_DICT['z']))
        conversions["z"] = DEFAULT_COSMO_DICT['z']

    conversions["omega_m"] = conversions["omega_b"] + \
        conversions["omega_cdm"] + conversions["omega_nu"]

    # Fill in default values for density parameters, because we need these to
    # compute h.

    # The question is, when omega_K is not specified, should we immediately set
    # it to default, or immediately set h to default and back-calculate
    # curvature?
    if "omega_DE" in conversions and "omega_K" not in conversions:
        if "h" not in conversions:
            conversions["omega_K"] = DEFAULT_COSMO_DICT["omega_K"]
        else:
            conversions["omega_K"] = conversions["h"] ** 2 - \
                conversions["omega_m"] - conversions["omega_DE"]
    elif "omega_K" not in conversions: # omega_DE also not in conversions
        conversions["omega_K"] = DEFAULT_COSMO_DICT["omega_K"]
        if "h" not in conversions:
            conversions["h"] = DEFAULT_COSMO_DICT["h"]
        conversions["omega_DE"] = conversions["h"] ** 2 - \
            conversions["omega_m"] - conversions["omega_K"]
            
    # Analogous block for dark energy
    if "omega_K" in conversions and "omega_DE" not in conversions:
        if "h" not in conversions:
            conversions["omega_DE"] = DEFAULT_COSMO_DICT["omega_DE"]
        else:
            conversions["omega_DE"] = conversions["h"] ** 2 - \
                conversions["omega_m"] - conversions["omega_K"]
                
    # If h wasn't given, compute it now that we have all of the physical
    # densities.
    if "h" not in conversions:
        DEFAULT_COSMO_DICT["h"] = np.sqrt(conversions["omega_m"] + \
            cosmology["omega_DE"] + cosmology["omega_K"])
        
    for i in range(len(physical_keys)):
        phys_key = physical_keys[i]
        frac_key = fractional_keys[i]
        if frac_key not in conversions:
            conversions[frac_key] = \
                conversions[phys_key] / conversions["h"] ** 2    
        
    # package it up for brenda
    if "As" not in conversions:
        conversions["As"] = DEFAULT_COSMO_DICT["As"]

    conversions["Omega_m"] = conversions["omega_m"] / conversions["h"] ** 2
    conversions["de_model"] = "w0wa"
    # We'll probably want to change this at some point, especially to allow
    # stuff like the infamous Aletheia model 8.
    conversions["Om_EdE"] = False
    # Brenda lib doesn't distinguish nu from CDM
    conversions["omega_cdm"] += conversions["omega_nu"]
    conversions["Omega_cdm"] += conversions["Omega_nu"]
    conversions["sigma8"] = None
    
    cosmology = brenda.Cosmology()
    for key in cosmology.pars.keys():
        if key not in conversions:
            conversions[key] = cosmology.pars[key]
    
    # The omega_K field will be ignored, but remembered through h
    # z will be ignored by brenda, but used by this code.
    cosmology.pars = conversions
        
    return cosmology

