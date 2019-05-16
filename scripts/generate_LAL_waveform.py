import numpy as np
import lal
import lalsimulation as lalsim
from lal import MSUN_SI, MTSUN_SI, PC_SI, C_SI

#-----------------------------------------------------------------
def generate_LAL_modes(approximant, q, chi1z, chi2z, dt, M, \
    dist_mpc, f_low, f_ref, phi_ref=0, modes_list=None):

    distance = dist_mpc* 1.0e6 * PC_SI

    approxTag = lalsim.SimInspiralGetApproximantFromString(approximant)

    # component masses of the binary
    m1_kg =  M*MSUN_SI*q/(1.+q)
    m2_kg =  M*MSUN_SI/(1.+q)

    dictParams = lal.CreateDict()

    # If a list of modes is given, load only those and their (l,-m) counterparts
    if modes_list is not None:
        # First, create the 'empty' mode array
        ma=lalsim.SimInspiralCreateModeArray()
        for mode in modes_list:
            # add (l,m) and (l,-m) modes
            lalsim.SimInspiralModeArrayActivateMode(ma, mode[0], mode[1])
            lalsim.SimInspiralModeArrayActivateMode(ma, mode[0], -mode[1])
        lalsim.SimInspiralWaveformParamsInsertModeArray(dictParams, ma)

    lmax = 5
    hmodes = lalsim.SimInspiralChooseTDModes(phi_ref, dt, m1_kg, m2_kg, \
        0, 0, chi1z, 0, 0, chi2z, f_low, f_ref, distance, dictParams, lmax, \
        approxTag)

    t = np.arange(len(hmodes.mode.data.data)) * dt
    mode_dict = {}
    while hmodes is not None:
        mode_dict[(hmodes.l, hmodes.m)] = hmodes.mode.data.data
        hmodes = hmodes.next
    return t, mode_dict

approximant = 'NRHybSur3dq8'
dt = 1./4096        # sec
dist_mpc = 100      # Mpc
q = 4               # m1/m2
chi1z = 0
chi2z = 0
M = 80              # total mass in Solar masses
f_low = 20        # Hz
f_ref = 20        # Hz

# time, h32 = h_dict[(3,2)]
t, h_dict = generate_LAL_modes(approximant, q, chi1z, chi2z, dt, M, \
    dist_mpc, f_low, f_ref, modes_list=[(2,2),(2,1),(3,3),(4,4)])
print h_dict.keys()

# The first evaluation also loads the surrogate, which can be slow. But on
# subsequent evaluations we don't need to reload it, so it's faster. Use
# the second evaluation to test timing.
import time
start = time.time()
t, h_dict = generate_LAL_modes(approximant, q, chi1z, chi2z, dt, M, \
    dist_mpc, f_low, f_ref, modes_list=[(2,2),(2,1),(3,3),(4,4)])
np.fft.fft(h_dict[(2,2)])
end = time.time()
print end-start
