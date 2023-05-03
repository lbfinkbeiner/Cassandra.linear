path_base_linux = "/home/lfinkbei/Documents/"
path_base_rex = "C:/Users/Lukas/Documents/GitHub/"
path_base_otto = "T:/GitHub/"
path_base = path_base_linux

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import camb
import re
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

'''Keep in mind that this is NOT the same file as the original
"cosmology_Aletheia.dat" that Ariel gave us! If you use the unaltered version,
you will get a segfault'''
path_to_me = path_base + "Master/CAMB_notebooks/shared/"
cosm = pd.read_csv(path_to_me + "data/cosmologies.dat", sep='\s+')

omegas_nu = np.array([0.0006356, 0.002148659574468, 0.006356, 0.01])
# Add corresponding file accessors, to check our work later
omnu_strings = np.array(["0.0006", "0.002", "0.006", "0.01"])

# The following code is somewhat hard;
# I'm not sure how better to do it.
redshift_column = re.compile("z.+")

'''! We really ought to merge the next three functions'''

def define_powernu(relative_path, omeganu_strings=None):
    file_base = file_base = path_to_me + relative_path

    def iterate_over_models_and_redshifts(accessor="0.002"):
        nested_spectra = []
        for i in range(0, 9): # iterate over models
            nested_spectra.append([])
            for j in range(0, 5): # iterate over snapshots
                nested_spectra[i].append(pd.read_csv(file_base + \
                    accessor + "_caso" + str(i) + "_000" + str(j) + ".dat",
                    names=["k", "P_no", "P_nu", "ratio"], sep='\s+'))

        return nested_spectra

    if omeganu_strings is None:
        return iterate_over_models_and_redshifts()

    powernu = {}

    for i in range(len(omeganu_strings)):
        file_accessor = omeganu_strings[i]
        powernu_key = omnu_strings[i]

        powernu[powernu_key] = \
            iterate_over_models_and_redshifts(file_accessor)

    return powernu

### Just some standard colors and styles for when I plot several models
# together.
colors = ["green", "blue", "brown", "red", "black", "orange", "purple",
          "magenta", "cyan"] * 200

#styles = ["solid", "dotted", "dashed", "dashdot", "solid", "dotted", "dashed",
#    "dashdot"]
# Line styles are unfortunately too distracting in plots as dense as those with
# which we are here dealing; make everything solid
styles = ["solid"] * 200

def is_matchable(target, cosmology):
    # I thought bigger sigma12 values were supposed to come with lower z,
    # but the recent matching results have got me confused.
    _, _, _, s12_big = kzps(cosmology, 0, nu_massive=False, zs=[0])

def match_s12(target, tolerance, cosmology,
    _redshifts=np.flip(np.linspace(0, 1100, 150)), _min=0):
    """
        Return a redshift at which to evaluate the power spectrum of cosmology
    @cosmology such that the sigma12_massless value of the power spectrum is
    within @tolerance (multiplicative discrepancy) of @target.
 
    @target this is the value of sigma12_massless at the assumed redshift
        (e.g. typically at z=2.0 for a standard Aletheia model-0 setup).
    @cosmology this is the cosmology for which we want to find a sigma12 value,
        so this will typically be an exotic or randomly generated cosmology
    @tolerance ABS((target - sigma12_found) / target) <= tolerance is the
        stopping condition for the binary search that this routine uses.
    @_z: the redshift to test at. This is part of the internal logic of the
        function and should not be referenced elsewhere.
    """
    # If speed is an issue, let's reduce the k samples to 300, or at least
    # add num_samples as a function parameter to kzps

    # First, let's probe the half-way point.
    # We're assuming a maximum allowed redshift of $z=2$ for now.

    #print(_z)
    _, _, _, list_s12 = kzps(cosmology, 0, nu_massive=False, zs=_redshifts)

    import matplotlib.pyplot as plt
    #print(list_s12)
    if False:
        plt.plot(_redshifts, list_s12);
        plt.axhline(sigma12_in)
        plt.show()
     
    # debug block
    if False:
        plt.plot(_redshifts, list_s12 - sigma12_in);
        plt.axhline(0)
        plt.show()
    

    list_s12 -= target # now it's a zero-finding problem

    # For some reason, flipping both arrays helps the interpolator
    # But I should come back and check this, I'm not sure if this was just a
    # patch for the Newton method
    interpolator = interp1d(np.flip(_redshifts), np.flip(list_s12),
        kind='cubic')
    try:
        z_best = root_scalar(interpolator, bracket=(np.min(_redshifts),
            np.max(_redshifts))).root
    except ValueError:
        print("No solution.")
        return None # there is no solution

    _, _, _, s12_out = kzps(cosmology, 0, nu_massive=False, zs=[z_best])
    discrepancy = (s12_out[0] - target) / target 
    if abs(discrepancy) <= tolerance:
        return z_best
    else:
        z_step = _redshifts[0] - _redshifts[1]
        new_floor = max(0, z_best - z_step)
        new_ceil = min(1100, z_best + z_step)
        return match_s12(target, tolerance, cosmology, _redshifts = \
            np.flip(np.linspace(new_floor, new_ceil, 150)))

def get_As_matched_cosmology(A_s=2.12723788013000E-09):
    """
    Unfortunately, all of these bounds are hard-coded. Maybe we can read in a
    table for this?

    The default A_s value corresponds to model 0

    Warning! You may have to throw out some of the cosmologies that you get
    from this routine because I am nowhere guaranteeing that the sigma12 you
    want actually corresponds to a positive redshift... of course, this
    wouldn't be a problem if CAMB allowed negative redshifts.
    """
    row = {}

    # Shape parameters: CONSTANT ACROSS MODELS
    row['ombh2'] = 0.022445
    row['omch2'] = 0.120567
    row['n_s'] = 0.96

    row['h'] = np.random.uniform(0.2, 1)
   
    # Given h, the following are now fixed:
    row['OmB'] = row['ombh2'] / row['h'] ** 2
    row['OmC'] = row['omch2'] / row['h'] ** 2
    row['OmM'] = row['OmB'] + row['OmC'] 

    row['OmK'] = 0
    row['OmL'] = 1 - row['OmM'] - row['OmK']
    
    #~ Do we have any constraints on h besides Aletheia?
    # I ask because this seems like a pretty small window.
    #ditto
    row['w0'] = np.random.uniform(-2., -.5)
    # ditto
    row['wa'] = np.random.uniform(-0.5, 0.5)

    row['A_s'] = A_s

    return row

def get_random_cosmology():
    """
    Unfortunately, all of these bounds are hard-coded. Maybe we can read in a
    table for this?
    """
    row = {}

    # Shape parameters: CONSTANT ACROSS MODELS
    row['ombh2'] = 0.022445
    row['omch2'] = 0.120567
    row['n_s'] = 0.96

    row['h'] = np.random.uniform(0.2, 1)
   
    # Given h, the following are now fixed:
    row['OmB'] = row['ombh2'] / row['h'] ** 2
    row['OmC'] = row['omch2'] / row['h'] ** 2
    row['OmM'] = row['OmB'] + row['OmC'] 

    row['OmK'] = np.random.uniform(-0.05, 0)
    row['OmL'] = 1 - row['OmM'] - row['OmK']
    
    #~ Do we have any constraints on h besides Aletheia?
    # I ask because this seems like a pretty small window.
    #ditto
    row['w0'] = np.random.uniform(-2., -.5)
    # ditto
    row['wa'] = np.random.uniform(-0.5, 0.5)

    A_min = np.exp(1.61) / 10 ** 10
    A_max = np.exp(5) / 10 ** 10
    row['A_s'] = np.random.uniform(A_min, A_max)

    #~ Should we compute omnuh2 here, or leave that separate?

    #~ Should we also specify parameters not specified by the Aletheia data
        # table, for example tau or the CMB temperature?

    return row
        
def boltzmann_battery(onh2s, onh2_strs, skips_omega = [0, 2],
    skips_model=[8], skips_snapshot=[1, 2, 3], h_units=False,
    models=cosm, fancy_neutrinos=False):
    """
    Return format uses an architecture that closely agrees with that of Ariel's
    in the powernu results:
    spec_sims
        omnuh2 str
            model index
                snapshot index
                    quantity of interest

    Although this agreement is an added benefit, the main point is simply to
    have a cleaner and more versatile architecture than the mess of separate
    arrays returned previously. So even if the "ground truth" object should
    eventually cease to agree in shape, this function already returns a much
    more pleasant object.

    Another difference with boltzmann_battery: this function automatically
    uses h_units=False, which should further bring my object into agreement
    with powernu. This is more debatable than simple architecture cleanup, so I
    will leave this as a flag up to the user.
    """
    assert type(onh2s) == list or type(onh2s) == np.ndarray, \
        "if you want only one omega value, you must still nest it in a list"
    assert type(onh2_strs) == list or type(onh2_strs) == np.ndarray, \
        "if you want only one omega value, you must still nest it in a list"
    assert len(onh2s) == len(onh2_strs), "more or fewer labels than points"
    
    spec_sims = {}

    for om_index in range(len(onh2s)):
        print(om_index % 10, end='')
        om = onh2s[om_index]
        om_str = onh2_strs[om_index]
        if om_index in skips_omega:
            spec_sims[om_str] = None
            continue
        spec_sims[om_str] = []
        for mindex, row in models.iterrows():
            if mindex in skips_model:
                # For example, I don't yet understand how to implement model 8
                spec_sims[om_str].append(None)
                continue
                
            h = row["h"]
            spec_sims[om_str].append([])
       
            z_input = parse_redshifts(mindex)
            if None in z_input:
                spec_sims[om_str][m_index] = None
                continue

            #print("z_input", z_input)
            #print("total Zs", len(z_input)) 
            for snap_index in range(len(z_input)):
                '''
                since z_input is ordered from z large to z small,
                and snap indices run from z large to z small,
                z_index = snap_index in this case and NOT in general
                '''
                #print(z_index)
                if snap_index in skips_snapshot:
                    #print("skipping", z_index)
                    spec_sims[om_str][mindex].append(None)
                    continue
                #print("using", z_index)
                inner_dict = {}
                z = z_input[snap_index]
              
                massless_tuple = kzps(row, om, nu_massive=False, zs=[z],
                    fancy_neutrinos=fancy_neutrinos)
                inner_dict["k"] = massless_tuple[0] if h_units \
                    else massless_tuple[0] * h
                inner_dict["P_no"] = massless_tuple[2] if h_units \
                    else massless_tuple[2] / h ** 3
                inner_dict["s12_massless"] = massless_tuple[3]

                massive_tuple = kzps(row, om, nu_massive=True, zs=[z],
                    fancy_neutrinos=fancy_neutrinos)
                inner_dict["P_nu"] = massive_tuple[2] if h_units \
                    else massive_tuple[2] / h ** 3
                inner_dict["s12_massive"] = massive_tuple[3]
                
                # Temporary addition, for debugging
                inner_dict["z"] = z_input[snap_index]               
 
                assert np.array_equal(massless_tuple[0], massive_tuple[0]), \
                   "assumption of identical k axes not satisfied!"
                    
                spec_sims[om_str][mindex].append(inner_dict) 

    return spec_sims

def kzps(mlc, omnuh2_in, nu_massive=False, zs = [0], nnu_massive_in=1,
    fancy_neutrinos=False):
    """
    Returns the scale axis, redshifts, power spectrum, and sigma12
    of a Lambda-CDM model
    @param mlc : "MassLess Cosmology"
        a dictionary of values
        for CAMBparams fields
    @param omnuh2_in : neutrino physical mass density
    @nu_massive : if this is True,
        the value in omnuh2_in is used to set omnuh2.
        If this is False,
        the value in omnuh2_in is simply added to omch2.

    Possible mistakes:
    A. We're setting "omk" with OmK * h ** 2. Should I have used OmK? If so,
        the capitalization here is nonstandard.
    """ 

    pars = camb.CAMBparams()
    omch2_in = mlc["omch2"]

    mnu_in = 0
    nnu_massive = 0
    h = mlc["h"]

    if nu_massive:
        '''This is a horrible workaround, and I would like to get rid of it
        ASAP. It destroys dependence on TCMB and
        neutrino_hierarchy, possibly more. But CAMB does not accept omnuh2 as
        an input, so I have to reverse-engineer it somehow.
        
        Also, should we replace default_nnu with something else in the
        following expression? Even if we're changing N_massive to 1,
        N_total_eff = 3.046 nonetheless, right?'''
        mnu_in = omnuh2_in * camb.constants.neutrino_mass_fac / \
            (camb.constants.default_nnu / 3.0) ** 0.75 
        #print("The mnu value", mnu_in, "corresponds to the omnuh2 value",
        #    omnuh2_in)
        omch2_in -= omnuh2_in
        nnu_massive = nnu_massive_in

    # tau is a desperation argument
    pars.set_cosmology(
        H0=h * 100,
        ombh2=mlc["ombh2"],
        omch2=omch2_in,
        omk=mlc["OmK"],
        mnu=mnu_in,
        #num_massive_neutrinos=nnu_massive, CODE_BLUE
        tau=0.0952, # just like in Matteo's notebook, at least (but maybe I got
            # this value from somewhere else...)
        neutrino_hierarchy="degenerate" # 1 eigenstate approximation; our
        # neutrino setup (see below) is not valid for inverted/normal
        # hierarchies.
    )
    # Matteo really didn't use any of this block? I don't understand--this
    # block seems well theoretically motivated.
    if fancy_neutrinos:
        pars.num_nu_massless = 3.046 - nnu_massive
        pars.nu_mass_eigenstates = nnu_massive
        stop_i = pars.nu_mass_eigenstates + 1
        pars.nu_mass_numbers[:stop_i] = \
            list(np.ones(len(pars.nu_mass_numbers[:stop_i]), int))
        pars.num_nu_massive = 0
        if nnu_massive != 0:
            pars.num_nu_massive = sum(pars.nu_mass_numbers[:stop_i])
    
    pars.InitPower.set_params(As=mlc["A_s"], ns=mlc["n_s"],
        r=0, nt=0.0, ntrun=0.0) # the last three are desperation arguments
    
    ''' The following seven lines are desperation settings
    If we ever have extra time, we can more closely study what each line does
    '''
    # This is a desperation line in light of the previous line. The previous
    # line seems to have served me well enough so far, but BSTS.
    pars.NonLinear = camb.model.NonLinear_none
    pars.WantCls = False
    pars.WantScalars = False
    pars.Want_CMB = False
    pars.DoLensing = False
    pars.YHe = 0.24
    # Matteo used this line but Andrea uses the following lines
    pars.set_accuracy(AccuracyBoost=2)
    # I already verified that commenting-out the next four lines DOES NOT
    # impact the Lukas-Matteo Gap.
    pars.Accuracy.AccuracyBoost = 3
    pars.Accuracy.lAccuracyBoost = 3
    pars.Accuracy.AccuratePolarization = False
    pars.Transfer.kmax = 20.0 / h

    # desperation if statement
    # CODE_GREEN should we add the additional try/catch that Matteo uses?
    if mlc["w0"] != -1 or float(mlc["wa"]) != 0:
        pars.set_dark_energy(w=mlc["w0"], wa=float(mlc["wa"]),
            dark_energy_model='ppf')
    
    ''' To change the the extent of the k-axis, change the following line as
    well as the "get_matter_power_spectrum" call. '''
    pars.set_matter_power(redshifts=zs, kmax=20.0 / h, nonlinear=False)
    
    results = camb.get_results(pars)

    # WHY DOESN'T THIS MAKE ANY DIFFERENCE???
    #results.calc_power_spectra(pars)

    sigma12 = results.get_sigmaR(12, hubble_units=False)
    
    '''
    In some cursory tests, the accurate_massive_neutrino_transfers
    flag did not appear to significantly alter the outcome.
    
    The flags var1=8 and var2=8 indicate that we are looking at the
    power spectrum of CDM + baryons (i.e. neutrinos excluded).
    '''
    k, z, p = results.get_matter_power_spectrum(
        minkh=1e-4 / h, maxkh=10.0 / h, npoints = 100000,
        var1=8, var2=8
    )
   
    # De-nest for the single-redshift case:
    if len(p) == 1:
        p = p[0] 
    return k, z, p, sigma12 

def model_ratios(snap_index, sims, canvas, massive=True, skips=[],
    subplot_indices=None, active_labels=['x', 'y'], title="Ground truth",
    omnuh2_str="0.002", models=cosm, suppress_legend=False):
    """
    Why is this a different function from above?
    There are a couple of annoying formatting differences with the power nu
    dictionary which add up to an unpleasant time trying to squeeze it into the
    existing function...

    Here, the baseline is always model 0,
    but theoretically it should be quite easy
    to generalize this function further.
    """
    P_accessor = None
    if massive == True:
         P_accessor = "P_nu"
    elif massive==False:
        P_accessor = "P_no"
 
    baseline_h = models.loc[0]["h"]
    baseline_k = correct_sims[0][snap_index]["k"]
    
    baseline_p = sims[0][snap_index]["P_nu"] / \
        sims[0][snap_index]["P_no"]
    if P_accessor is not None:
        baseline_p = sims[0][snap_index][P_accessor]
    
    plot_area = canvas # if subplot_indices is None
    if subplot_indices is not None:
        if type(subplot_indices) == int:
            plot_area = canvas[subplot_indices]
        else: # we assume it's a 2d grid of plots
            plot_area = canvas[subplot_indices[0], subplot_indices[1]]
        # No need to add more if cases because an n-d canvas of n > 2 makes no
        # sense.
    
    k_list = []
    rat_list = []
    for i in range(1, len(correct_sims)):
        if i in skips:
            continue # Don't know what's going on with model 8
        this_h = models.loc[i]["h"]
        this_k = correct_sims[i][snap_index]["k"]
        
        this_p = correct_sims[i][snap_index]["P_nu"] / \
            correct_sims[i][snap_index]["P_no"]
        if P_accessor is not None:
            this_p = correct_sims[i][snap_index][P_accessor]
        
        truncated_k, truncated_p, aligned_p = \
            truncator(baseline_k, baseline_p, this_k,
                this_p, interpolation=True)

        label_in = "model " + str(i)
        plot_area.plot(truncated_k, aligned_p / truncated_p,
                 label=label_in, c=colors[i], linestyle=styles[i])
       
        k_list.append(truncated_k)
        rat_list.append(aligned_p / truncated_p)
 
    plot_area.set_xscale('log')
    if 'x' in active_labels:
        plot_area.set_xlabel(r"k [1 / Mpc]")
   
    ylabel =  r"$x_i / x_0$"
    if P_accessor is not None:
        if massive == True:
            ylabel = r"$P_\mathrm{massive} / P_\mathrm{massive, model \, 0}$"
        if massive == False:
            ylabel = r"$P_\mathrm{massless} / P_\mathrm{massless, model \, 0}$"
    
    if 'y' in active_labels:
        plot_area.set_ylabel(ylabel)
    
    plot_area.set_title(title + r": $\omega_\nu$ = " + omnuh2_str + \
        "; Snapshot " + str(snap_index))
    if not suppress_legend:
        plot_area.legend()

    return k_list, rat_list

def compare_wrappers(k_list, p_list, correct_sims, snap_index,
    canvas, massive, subscript, title, skips=[], subplot_indices=None,
    active_labels=['x', 'y']):
    """
    Python-wrapper (i.e. Lukas') simulation variables feature the _py ending
    Fortran (i.e. Ariel's) simulation variables feature the _for ending
    """
    
    P_accessor = None
    if massive == True:
        P_accessor = "P_nu"
    elif massive == False:
        P_accessor = "P_no"
    x_mode = P_accessor is None

    # Remember, the returned redshifts are in increasing order
    # Whereas snapshot indices run from older to newer
    z_index = 4 - snap_index

    baseline_h = cosm.loc[0]["h"]
    
    baseline_k_py = k_list[0] * baseline_h
    
    baseline_p_py = None
    if x_mode:
        baseline_p_py = p_list[0][z_index]
    else:
        baseline_p_py = p_list[0][z_index] / baseline_h ** 3
    
    baseline_k_for = correct_sims[0][snap_index]["k"]
    
    baseline_p_for = correct_sims[0][snap_index]["P_nu"] / \
        correct_sims[0][snap_index]["P_no"]
    if P_accessor is not None:
        baseline_p_for = correct_sims[0][snap_index][P_accessor]
    
    plot_area = None
    if subplot_indices is None:
        plot_area = canvas
    elif type(subplot_indices) == int:
        plot_area = canvas[subplot_indices]
    else:
        plot_area = canvas[subplot_indices[0], subplot_indices[1]]

    # k_list is the LCD because Ariel has more working models than I do
    for i in range(1, len(k_list)):
        if i in skips:
            continue
        this_h = cosm.loc[i]["h"]
        
        this_k_py = k_list[i] * this_h
        this_p_py = None
        if x_mode==False:
            this_p_py = p_list[i][z_index] / this_h ** 3
        else:
            this_p_py = p_list[i][z_index]

        this_k_for = correct_sims[i][snap_index]["k"]
        
        this_p_for = correct_sims[i][snap_index]["P_nu"] / \
            correct_sims[i][snap_index]["P_no"]
        if P_accessor is not None:
            this_p_for = correct_sims[i][snap_index][P_accessor]

        truncated_k_py, truncated_p_py, aligned_p_py = \
            truncator(baseline_k_py, baseline_p_py, this_k_py,
                this_p_py, interpolation=this_h != baseline_h)
        y_py = aligned_p_py / truncated_p_py

        truncated_k_for, truncated_p_for, aligned_p_for = \
            truncator(baseline_k_for, baseline_p_for, this_k_for,
            this_p_for, interpolation=this_h != baseline_h)
        y_for = aligned_p_for / truncated_p_for

        truncated_k, truncated_y_py, aligned_p_for = \
            truncator_neutral(truncated_k_py, y_py, truncated_k_for, y_for) 

        label_in = "model " + str(i)
        plot_area.plot(truncated_k,
            truncated_y_py / aligned_p_for, label=label_in, c=colors[i],
            linestyle=styles[i])

    plot_area.set_xscale('log')
    if 'x' in active_labels:
        plot_area.set_xlabel(r"k [1 / Mpc]")
    
    ylabel = None
    if x_mode:
        ylabel = r"$ж_i/ж_0$"
    else:
        ylabel = r"$y_\mathrm{py} / y_\mathrm{fortran}$"
    
    if 'y' in active_labels:
        plot_area.set_ylabel(ylabel)
 
    plot_area.set_title(title)
    plot_area.legend()

    plot_area.set_title(title)
    plot_area.legend()

def parse_redshifts(model_num):
    """
    Return the list of amplitude-equalized redshifts
    given for a particular model in the Aletheia dat file.
    
    This function is intended to return the redshifts
    in order from high (old) to low (recent),
    which is the order that CAMB will impose
    if not already used.
    """
    z = []
    try:
        model = cosm.loc[model_num]
    
        for column in cosm.columns:
            if redshift_column.match(column):
                z.append(model[column])
    except (ValueError, KeyError):
        z = [3, 2, 1, 0]
            
    # No need to sort these because they should already
    # be in descending order.
    return np.array(z)

def truncator(base_x, base_y, obj_x, obj_y):
    # This doesn't make the same assumptions as truncator
    # But of course it's terrible form to leave both of these functions here.
    """
    Throw out base_x values until
        min(base_x) >= min(obj_x) and max(base_x) <= max(obj_x)    
    then interpolate the object arrays over the truncated base_x domain.
    @returns:
        trunc_x: truncated base_x array, which is now common to both y arrays
        trunc_y: truncated base_y array
        aligned_y: interpolation of obj_y over trunc_x
    """
    # What is the most conservative lower bound?
    lcd_min = max(min(obj_x), min(base_x))
    # What is the most conservative upper bound?
    lcd_max = min(max(obj_x), max(base_x))
    
    # Eliminate points outside the conservative bounds
    mask_base = np.all([[base_x <= lcd_max], [base_x >= lcd_min]], axis=0)[0]
    trunc_base_x = base_x[mask_base]
    trunc_base_y = base_y[mask_base]
   
    mask_obj = np.all([[obj_x <= lcd_max], [obj_x >= lcd_min]], axis=0)[0]
    trunc_obj_x = obj_x[mask_obj]
    trunc_obj_y = obj_y[mask_obj]
 
    interpolator = interp1d(obj_x, obj_y, kind="cubic")
    aligned_y = interpolator(trunc_base_x)

    #print(len(trunc_base_x), len(aligned_y)) 
    return trunc_base_x, trunc_base_y, aligned_y
