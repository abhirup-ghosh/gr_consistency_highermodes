#############################################################################
##
##      Filename: generate_NRHybSur3dq8.py
##
##      Author: Vijay Varma
##
##      Created: 28-02-2019
##
##      Description: Wrapper for NRHybSur3dq8 model
##
#############################################################################

import numpy as np
import gwsurrogate

def eval_waveform(sur, q, M, chi1z, chi2z, dt, f_low, dist_mpc, inclination, \
        phi_ref, mode_list):
    """ Evaluates NRHybSur3dq8 waveform model.

    Input:
        sur         : Preloaded surrogate object. ONLY NEEDS TO BE LOADED ONCE.
        q           : Mass ratio, m1/m2 >= 1.
        M           : Total mass in solar masses.
        chi1z       : Dimensionless spin of heavier BH.
        chi2z       : Dimensionless spin of lighter BH.
        dt          : Time step in seconds.
        f_low       : Initial frequency in Hz.
        dist_mpc    : Distance in MegaParsecs.
        inclination : Inclination angle b/w orbital angular momentum direction
                      and line-of-sight to detector.
        phi_ref     : Initial orbital phase.
        mode_list   : List of modes to evaluate. Example: [(2,2), (2,1)]. The
                      m<0 modes are automatically added.

    Output:
        t           : Time values in seconds.
        h           : Complex strain h_{+} -1j * h_{x}.


    ##### Usage instructions

    ### Installation

    pip install gwsurrogate

    Or get from source (https://pypi.org/project/gwsurrogate/)

    ### Downloading surrogate data (This only needs to be done once, ever)

    import gwsurrogate
    # This can take a few minutes
    gwsurrogate.catalog.pull('NRHybSur3dq8')

    ### Loading surrogate

    # This only needs to be done once at the start of your script
    sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')

    ### Evaluation

    q = 4
    M = 60             # Total masss in solar masses
    chi1z = 0.5
    chi2z = -0.7
    dist_mpc = 100     # distance in megaparsecs
    dt = 1./4096       # step size in seconds
    f_low = 20         # initial frequency in Hz
    inclination = np.pi/4
    phi_ref = np.pi/5

    # Modes to include. The m<0 modes are automatically added.
    mode_list = [(2,1)]
    t, h = eval_waveform(sur, q, M, chi1z, chi2z, dt, f_low, dist_mpc, \
        inclination, phi_ref, mode_list)

    import matplotlib.pyplot as P
    P.plot(t, h.real, label='$h_{+}$ $(\iota=%.2f, \phi_{ref}=%.2f)$'%( \
            inclination, phi_ref))
    P.plot(t, -h.imag, label='$h_{\\times}$ $(\iota=%.2f, \phi_{ref}=%.2f)$'%( \
            inclination, phi_ref))
    P.legend(fontsize=14)
    P.xlabel('t [s]', fontsize=14)
    P.show()
    """

    x = [q, chi1z, chi2z]
    t, h = sur(x, dt=dt, f_low=f_low, mode_list=mode_list, M=M, \
        dist_mpc=dist_mpc, inclination=inclination, phi_ref=phi_ref, \
        units='mks')
    return t, h
