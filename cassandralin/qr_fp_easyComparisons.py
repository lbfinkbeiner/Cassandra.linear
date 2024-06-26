# coding: utf-8
import test_cases

import numpy as np
lhc = np.load("tests/lhc_massless_fp.npy")
true = np.load("tests/Feb16masslessWiggle.npy")
lil_k = np.load("300k.npy")
from cassL import user_interface as ui
from cassL import generate_emu_data as ged
priors = ui.prior_file_to_array("COMET_PLUS_FP")
mapping = ged.labels_to_mapping(ged.COSMO_PARS_INDICES)
errors, predictions = test_cases.easy_comparisons(
        lhc, true, priors, lil_k, mapping)
