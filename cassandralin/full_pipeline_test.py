import numpy as np

from cassL import generate_emu_data as ged
from cassL import user_interface as ui

lhc = np.load("lhc_full_pipeline_test.npy")[:200]
priors = ui.prior_file_to_array("COMET_FP")
k_axis = np.load("300k.npy") # this is actually 300k2, just renamed

import test_cases
import emulator_interface as ei

predictions = np.zeros((len(lhc), len(k_axis)))

for i in range(len(lhc)):
    print(i)
    row = lhc[i]
    ci_cosmology = ged.build_cosmology(ged.denormalize_row(row, priors))
    cosmo_dict = test_cases.ci_to_cosmodict(ci_cosmology)
    # val is trash, it's not useful
    intrpr, val = ei.get_Pk_interpolator(cosmo_dict)
    predictions[i] = intrpr(k_axis)

def get_errors(true, predictions):
    errors = np.zeros((len(true), len(k_axis)))
    for i in range(len(true)):
        true_spec = true[i]
        pred_spec = predictions[i]
        errors[i] = (true_spec - pred_spec) / true_spec
    return errors
    
