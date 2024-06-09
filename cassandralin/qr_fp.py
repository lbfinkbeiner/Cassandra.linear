get_ipython().run_line_magic('run', 'test_cases')

from cassL import camb_interface as ci
from cassL import generate_emu_data as ged
from cassL import user_interface as ui
import numpy as np
import matplotlib.pyplot as plt

priors = ui.prior_file_to_array("COMET_PLUS_FP")
lhc = np.load("tests/13Feb_lhc.npy")

denormd = ged.denormalize_row(lhc[3], priors)
cosm = ged.build_cosmology(denormd)
z = cosm['z']

redshifts = np.flip(np.linspace(max(0, z - 1), z + 1, 150))
intrpr = ci.cosmology_to_PK_interpolator(cosm, redshifts, kmax=7)

lil_k = np.load("300k.npy")

Pk = intrpr(z, lil_k)[0]

plt.loglog(lil_k, Pk)
plt.show()
