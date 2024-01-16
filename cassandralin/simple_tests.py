### The point of this script:
# Verify that curvature is handled correcttly by our sigma12-rescaling code!

from cassL import camb_interface as ci
import emulator_interface as ei
import numpy as np

m0 = ci.full_cosm.iloc[0]

def alli_to_ei(alli_cosm, z):
    conversions = ei.transcribe_cosmology(omB=alli_cosm["ombh2"],
                                       omC=alli_cosm["omch2"],
                                       n_s=alli_cosm["n_s"],
                                       A_s=alli_cosm["A_s"],
                                       OmK=alli_cosm["OmK"],
                                       h=alli_cosm["h"], z=z)
    return conversions


def it(conversions):
    omM = conversions["omB"] + conversions["omC"]
    OmM = omM / conversions["h"] ** 2
    OmK = conversions["omK"] / conversions["h"] ** 2
    OmDE = conversions["omDE"] / conversions["h"] ** 2
    LGF = ei.linear_growth_factor(OmM, OmK, OmDE, conversions["z"])
    
    return LGF


# Get some true answers about the D ratios

# z_values = [4, 2, 1, 0.5, 0.25, 0]
z_values = [3, 0.5, 0]

import copy as cp
m0_small = ci.default_cosmology()
m1 = cp.deepcopy(m0_small)
m1["OmK"]=0.05

m2 = cp.deepcopy(m0_small)
m2["OmK"]=-0.05

k0, z0, p0, s0 = ci.evaluate_cosmology(m0_small, z_values)
k1, z1, p1, s1 = ci.evaluate_cosmology(m1, z_values)
k2, z2, p2, s2 = ci.evaluate_cosmology(m2, z_values)

import matplotlib.pyplot as plt

for i in range(len(z_values)):
    z = z_values[i]
    plt.loglog(k0, p0[i] / p1[i])
    plt.title("01, z= " + str(z))
    plt.show()

    plt.loglog(k0, p0[i] / p2[i])
    plt.title("02, z= " + str(z))
    plt.show()

er01 = []
er02 = []

for z in z_values:
    m0_conv = alli_to_ei(m0, z)
    m1_conv = alli_to_ei(m1, z)
    m2_conv = alli_to_ei(m2, z)
    D0 = it(m0_conv)
    D1 = it(m1_conv)
    D2 = it(m2_conv)
    er01.append(D0 / D1 * np.sqrt(m0["A_s"] / m1["A_s"]))
    er02.append(D0 / D2 * np.sqrt(m0["A_s"] / m2["A_s"]))

print(er01)
print(er02)

# z = 2, z = 1
# 1.09747, 1.07675 (+0.05)
# 0.907127, 0.92552 (-0.05)

# z = 3, z = 0.5
# 1.0972, 1.08905
# .924008, 0.914086
