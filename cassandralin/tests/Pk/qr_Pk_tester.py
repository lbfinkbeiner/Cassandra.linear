import numpy as np
from cassandralin import emulator_interface as ei
from cassandralin import test_cases as tc
from cassL import user_interface as ui
from cassL import camb_interface as ci
from cassL import utils
import matplotlib.pyplot as plt

lil_k = np.load("../../300k.npy")

samples = np.load("Feb20_mini_fp_wiggles.npy")
lhc = np.load("lhc_full_pipeline.npy")
priors = ui.prior_file_to_array("COMET_PLUS_FP")

# When plotting the hi-res samples, you need to use big_k
big_k = np.load("65k_k.npy")
hi_res_samples = np.load("samples_backup_i0_through_9_fp_minimassive.npy")

qis = []
cds = []

for row in lhc[:10]:
    qi = tc.fetch_cosmology(row, priors)
    qis.append(qi)
    #MEMNeC = ci.balance_neutrinos_with_CDM(qi, 0)
    #print(ci.evaluate_sigma12(MEMNeC, [MEMNeC['z']]))
    cds.append(tc.ci_to_cosmodict_bare(qi))

def go(i):
    Pk_intrp, unc_intrp = ei.get_Pk_interpolator(cds[i])
    return Pk_intrp(lil_k)
    
def check(i):
    plt.loglog(lil_k, go(i), label="Ali-L")
    plt.loglog(lil_k, samples[i], label="CAMB")
    plt.xlabel("Scale $k$ [1 / Mpc]")
    plt.ylabel("P(k) [1 / Mpc$^3$]")
    plt.title("Comparison of CAMB and Ali-L Spectra")
    plt.legend()
    plt.show()
    
def check_at_hi_res(i):
    plt.loglog(lil_k, go(i), label="Ali-L")
    plt.loglog(big_k, hi_res_samples[i], label="CAMB hi-res")
    plt.xlabel("Scale $k$ [1 / Mpc]")
    plt.ylabel("P(k) [1 / Mpc$^3$]")
    plt.title("Comparison of CAMB and Ali-L Spectra")
    plt.legend()
    plt.show()
    
def error_curve(i):
    plt.plot(lil_k, utils.percent_error(samples[i], go(i)))
    plt.xscale('log')
    plt.xlabel("Scale $k$ [1 / Mpc]")
    plt.ylabel("% Error")
    plt.title("Comparison of CAMB and Ali-L Spectra")
    plt.show()
    