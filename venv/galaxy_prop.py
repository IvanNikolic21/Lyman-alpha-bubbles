import numpy as np
from astropy import units as u
from numpy.random import normal, binomial
from astropy.cosmology import z_at_value
from astropy.cosmology import Planck18 as Cosmo
from scipy import integrate
from scipy.stats import gaussian_kde
from astropy import constants as const

import py21cmfast as p21c

from venv.helpers import (
    wave_to_dv,
    gaussian,
    optical_depth,
    z_at_proper_distance,
    hmf_integral_gtm
)
from venv.igm_prop import get_bubbles, calculate_taus
from venv.igm_prop import calculate_taus_i

from venv.data.EndSta import get_ENDSTA_gals

from venv.chmf import chmf as chmf_func

wave_em = np.linspace(1214, 1225., 100) * u.Angstrom
dir_all = '/home/inikolic/projects/Lyalpha_bubbles/code/Lyman-alpha-bubbles/'

def delta_v_func(
        muv,
        z=7
):
    """
    Function returns velocity offset as a function of UV magnitude

    :param muv: float,
        UV magnitude of the galaxy.
    :param z: float,
        redshift of the galaxy.

    :return delta_v: float
        velocity offset of a given galaxy.
    """
    if muv >= -20.0 - 0.26 * z:
        gamma = -0.3
    else:
        gamma = -0.7
    return 0.32 * gamma * (muv + 20.0 + 0.26 * z) + 2.34


def get_js(
        muv=-18,
        z=7,
        n_iter=1,
        include_muv_unc=False,
        fwhm_true=False,
):
    """
    Function returns Lyman-alpha shape profiles

    :param muv: float,
        UV magnitude for which we're calculating the shape.
    :param z: float,
        redshift of interest.
    :param n_iter: integer,
        number of iterations of Lyman shape to get.
    :param include_muv_unc: boolean,
        whether to include the scatter in Muv.

    :return j_s: numpy.array of shape (N_iter, n_wav);
        array of profiles for a number of iterations and wavelengths.
    :return delta_vs : numpy.array of shape (N_iter);
        array of velocity offsets for a number of iterations.
    """
    n_wav = 100

    wv_off = wave_to_dv(wave_em)

    if hasattr(muv, '__len__'):
        tot_it_shape = (n_iter, *np.shape(muv))
    else:
        tot_it_shape = n_iter
    delta_vs = np.zeros(np.product(tot_it_shape))
    j_s = np.zeros((np.product(tot_it_shape), n_wav))

    if include_muv_unc and hasattr(muv, '__len__'):
        muv = np.array([np.random.normal(i, 0.1) for i in muv])
    elif include_muv_unc and not hasattr(muv, '__len__'):
        muv = np.random.normal(muv, 0.1)

    if hasattr(muv, '__len__'):
        delta_v_mean = np.array([delta_v_func(i, z) for i in muv.flatten()])
    else:
        delta_v_mean = delta_v_func(muv, z)

    for i in range(n_iter):
        if hasattr(muv, '__len__'):
            delta_vs[i] = 10**normal(delta_v_mean[i], 0.24)
        else:
            delta_vs[i] = 10**normal(delta_v_mean, 0.24)
        if fwhm_true:
            sigma=delta_vs[i] / 2 / np.sqrt(2 * np.log(2))
        else:
            sigma=delta_vs[i]
        j_s[i, :] = gaussian(wv_off.value, delta_vs[i], sigma)
        j_s[i, :] /= integrate.trapz(
            j_s[i, :],
            wave_em.value
        )

    if hasattr(muv, '__len__'):
        j_s.reshape((*tot_it_shape, n_wav))
        delta_vs.reshape(tot_it_shape)

    return j_s, delta_vs


def get_mock_data(
        n_gal=10,
        z_start=7.5,
        r_bubble=5,
        xb=0.0,
        yb=0.0,
        zb=0.0,
        dist=10,
        ENDSTA_data=False,
        diff_pos_prob=False,
):
    """

        Generate mock data
        Input:
        N_gal : integer,
            number of galaxies for which to create mock data.
        z_start : float,
            redshift of the center of the bubble.
        R_bubble : float,
            radius of the center ionized bubble.
        xb : float,
            x-position of the center of the main bubble.
        yb : float,
            y-position of the center of the main bubble.
        zb : float,
            z-position of the center of the main bubble.
        dist : float,
            distance up to which samples of galaxies are taken.
        ENDSTA_data : False
            whether to use data from Endsley & Stark 2021.
        diff_pos_prob : False
            whether galaxies have a different probability to be inside or out
            of the ionized region. Default is False.

        Returns:
        tau_data: numpy.array;
            List of transmissivities that represent the mock data.
        xs: numpy.array;
            x-coordinates of the mock galaxies.
        ys: numpy.array;
            y-coordinates of the mock galaxies.
        zs: numpy.array;
            z-coordinates of the mock galaxies. This is the LoS direction,
            negative values represent galaxies closer to us w.r.t. the center
            of the bubble.
        x_b: numpy.array;
            x-positions of the outside bubbles.
        y_b: numpy.array;
            y-positions of the outside bubbles.
        z_b: numpy.array;
            z-positions of the outside bubbles.
        r_bubs: numpy.array;
            radii of outside bubbles.
    """
    if ENDSTA_data:
        print("Using ENDSTA data")
        xs,ys,zs = get_ENDSTA_gals()
    else:
        if diff_pos_prob:
            xs = np.zeros(n_gal)
            ys = np.zeros(n_gal)
            zs = np.zeros(n_gal)
            for i in range(n_gal):
                success = False
                while not success:
                    x_try = np.random.uniform(-dist, dist,size=1)[0]
                    y_try = np.random.uniform(-dist, dist,size=1)[0]
                    z_try = np.random.uniform(-dist, dist,size=1)[0]
                    d_gal = np.sqrt(
                        (x_try-xb)**2 + (y_try-yb)**2 + (z_try-zb)**2
                    )
                    if d_gal < r_bubble:
                        if np.random.binomial(1,0.75):
                            success = True
                            xs[i] = x_try
                            ys[i] = y_try
                            zs[i] = z_try
                    else:
                        if np.random.binomial(1,0.25):
                            success = True
                            xs[i] = x_try
                            ys[i] = y_try
                            zs[i] = z_try
            #print("using this dist",dist,"and this radius", r_bubble, "this n_gal", n_gal, xs,ys,zs)
            #assert False

        else:
            xs = np.random.uniform(-dist, dist, size=n_gal)
            ys = np.random.uniform(-dist, dist, size=n_gal)
            zs = np.random.uniform(-dist, dist, size=n_gal)

    tau_data = np.zeros((n_gal, len(wave_em)))
    x_b, y_b, z_b, r_bubs = get_bubbles(0.65, 200, mock=True)
    #print(x_b,y_b,z_b,r_bubs)
    for i in range(n_gal):
        # red_s = z_at_value(
        #     Cosmo.comoving_distance,
        #     Cosmo.comoving_distance(z_start) + zs[i] * u.Mpc
        # )
        red_s = z_at_proper_distance(
            - zs[i] /(1+z_start) * u.Mpc, z_start
        )
        if (xs[i] - xb) ** 2 + (ys[i] - yb) ** 2 + (
                zs[i] - zb) ** 2 < r_bubble ** 2:
            dist = zs[i] - zb + np.sqrt(
                r_bubble ** 2 - (xs[i] - xb) ** 2 - (ys[i] - yb) ** 2)
            # z_end_bub = z_at_value(
            #     Cosmo.comoving_distance,
            #     Cosmo.comoving_distance(red_s) - dist * u.Mpc
            # )
            z_end_bub = z_at_proper_distance(
                dist / (1+red_s) * u.Mpc, red_s
            )
        else:
            dist = 0
            z_end_bub = red_s

        #Is it maybe different inconsistent taus?
        tau = calculate_taus_i(
            x_b,
            y_b,
            z_b,
            r_bubs,
            red_s,
            z_end_bub,
            n_iter=1,
            x_pos = xs[i],
            y_pos = ys[i],
            dist = dist
        )
        tau = np.nan_to_num(tau, np.inf)
        tau_data[i, :] = tau
    #print(x_b,y_b,z_b,r_bubs)
    #assert 1==0
    return tau_data, xs, ys, zs, x_b, y_b, z_b, r_bubs

def calculate_EW_factor(
        Muv,
        beta,
        mean=False,
        return_lum=True,
):
    """
    Function calculates the luminosity factor that is necessary to calculate
    equivalent width

    :param Muv: float
        UV magnitude of a given galaxy.
    :param beta: float
        a beta slope of SED.
    :param mean: boolean
        whether to just take the means from the distribution.
        Default is False
    :param return_flux: boolean;
        Do I return only the EW factor, or also the lyman-alpha luminosity.

    :return: EW_fac: float
        equivalent width that, when multiplied with transmission, gives EW.
    """

    Muv_thesan = np.load('/home/inikolic/projects/Lyalpha_bubbles/code/Lyman-alpha-bubbles/venv/data/Muv_THESAN.npy')
    La_thesan = np.load('/home/inikolic/projects/Lyalpha_bubbles/code/Lyman-alpha-bubbles/venv/data/Lya_THESAN.npy')
    if hasattr(Muv, '__len__'):
         La_sample = np.zeros((len(Muv)))
         La_sample_mean = np.zeros((len(Muv)))
         for i,(Muvi, beta_i) in enumerate(zip(Muv, beta)):
            
            Las = La_thesan[abs(Muv_thesan - Muvi) < 0.1]  # magnitude uncertainty
            #print(Las, Muv_thesan[abs(Muv_thesan - Muvi) < 0.1], Muvi, flush=True)
            if mean:
                La_sample[i] = np.mean(Las) * 3.846 * 1e33
            else:
                La_sample_mean[i] = np.mean(Las) * 3.846 * 1e33
                gk = gaussian_kde(Las * 3.846 * 1e33)
                La_sample[i] = gk.resample(1)[0][0]# * 3.846 * 1e33
    else:
        Las = La_thesan[abs(Muv_thesan - Muv) < 0.1] #magnitude uncertainty
        #print(Las, Muv_thesan[abs(Muv_thesan - Muv) < 0.1], Muv, flush=True)
        gk = gaussian_kde(Las)
        if mean:
            La_sample = np.mean(Las) * 3.846 * 1e33
        else:
            La_sample_mean = np.mean(Las) * 3.846 * 1e33
            gk = gaussian_kde(Las * 3.846 * 1e33)
            La_sample = gk.resample(1)# * 3.846 * 1e33

    L_UV_mean = 10**(-0.4*(Muv-51.6))
    C_const = 2.47 * 1e15 * u.Hz / 1216 / u.Angstrom * (1500 / 1216) ** (-beta-2)
    #print(C_const, flush=True)


    if return_lum:
        return La_sample / C_const.value / L_UV_mean, La_sample
    else:
        return La_sample / C_const.value / L_UV_mean


def get_muv(
        n_gal,
        redshift,
        muv_cut=-19.0,
        obs=False,
):
    """
    :param n_gal: integer,
        number of galaxies to sample from.
    :param redshift: float.
        redshift for lf.
    :param muv_cut: float,
        Muv cut. Default is -19.0
    :param obs: string or boolean,
        whether to use observed magnitudes
    :return: np.array of size ~ n_gal
        UV magnitudes of galaxies.
    """

    if obs == 'EndsleyStark21':
        return np.array([
            -21.5,
            -21.1,
            -21.3,
            -21.0,
            -21.3,
            -20.4,
            -21.5,
            -20.6,
            -20.7,
            -22.0,
            -21.2,
            -20.8,
        ])

    cosmo_params = {'hlittle': 0.6688, 'OMm': 0.321, 'OMb': 0.04952,
                    'POWER_INDEX': 0.9626, 'SIGMA_8': 0.8118}
    flag_options = {'USE_MASS_DEPENDENT_ZETA': True, "INHOMO_RECO": True,
                    "PHOTON_CONS": True, 'EVOLVING_R_BUBBLE_MAX': True}

    Map_parm = [
        -0.150343135732268118E+01,
        0.504878704249769550E+00,
        0.853431273778827304E+01,
        0.278575390476855700E+00,
        -0.971704175935387049E+00,
        -0.498862738992503996E+00,
    ]

    names = ['F_STAR10', 'ALPHA_STAR', 'M_TURN', 't_STAR', 'F_ESC10',
             'ALPHA_ESC']

    astro_params = dict(zip(names, Map_parm))

    Muv, mh, lf = p21c.compute_luminosity_function(
        astro_params=astro_params,
        cosmo_params=cosmo_params,
        flag_options=flag_options,
        redshifts=[redshift]
    )

    for i, m in enumerate(Muv[0][0:80]):
        if m < muv_cut:
            i = i - 1
            break
    Uvlf_cumsum = integrate.cumtrapz(10 ** lf[0][i:80], Muv[0][i:80])
    cumsum = Uvlf_cumsum / Uvlf_cumsum[-1]

    random_numb = np.random.uniform(size=n_gal)
    UV_list = np.zeros(shape=n_gal)

    for index, random in enumerate(random_numb):
        this_gal = 0.0
        while this_gal > muv_cut:
            this_gal = np.interp(
                random,
                np.concatenate((np.array([0.0]), cumsum)),
                np.array(Muv[0][i:80])
            )
            #print(this_gal)

            random = np.random.uniform(size=1)[0]
        UV_list[index] = this_gal

    return UV_list


def p_EW(
        Muv,
        beta=-2,
        mean=False,
        return_lum=True,
        high_prob_emit=False,
        EW_fixed=False
):
    """
    Function shall give sample from the distribution
    """

    def A(m):
        if high_prob_emit:
            return 0.95 + 0.05 * np.tanh(3 * (m+20.75))
        else:
            return 0.65 + 0.1 * np.tanh(3 * (m + 20.75))

    def W(m):
        return 31 + 12 * np.tanh(4 * (m + 20.25))

    if EW_fixed:
        if hasattr(beta, '__len__'):
            beta=beta[0]
        C_const = 2.47 * 1e15 * u.Hz / 1216 / u.Angstrom * (
                1500 / 1216) ** (-(beta) - 2)
        L_UV_mean = 10 ** (-0.4 * (Muv - 51.6))
        lum_alpha = W(Muv) * C_const.value * L_UV_mean

        return W(Muv), lum_alpha

    Ws = np.linspace(0, 500, 1000)

    if hasattr(Muv, '__len__') and not hasattr(beta, '__len__'):
        beta = beta * np.ones(len(Muv))

    C_const = 2.47 * 1e15 * u.Hz / 1216 / u.Angstrom * (1500 / 1216) ** (
                -(beta) - 2)
    L_UV_mean = 10 ** (-0.4 * (Muv - 51.6))

    if mean:
        if return_lum:
            return W(Muv) * A(Muv), W(Muv) * A(Muv) * C_const.value * L_UV_mean
        else:
            return W(Muv) * A(Muv)

    if hasattr(Muv, '__len__'):
        EWs = np.zeros((len(Muv)))
        if return_lum:
            lum_alpha = np.zeros((len(Muv)))
        for i, (muvi, beti) in enumerate(zip(Muv, beta)):
            if np.random.binomial(1, A(muvi)):
                EW_cumsum = integrate.cumtrapz(
                    1 / W(muvi) * np.exp(-Ws / W(muvi)), Ws)
                cumsum = EW_cumsum / EW_cumsum[-1]
                rn = np.random.uniform(size=1)
                EW_now = \
                np.interp(rn, np.concatenate((np.array([0.0]), cumsum)), Ws)[0]
            else:
                EW_now = 0.0
            EWs[i] = EW_now
            if return_lum:
                C_const = 2.47 * 1e15 * u.Hz / 1216 / u.Angstrom * (
                            1500 / 1216) ** (-(beti) - 2)
                L_UV_mean = 10 ** (-0.4 * (muvi - 51.6))
                lum_alpha[i] = EW_now * C_const.value * L_UV_mean
        if return_lum:
            return EWs, lum_alpha
        else:
            return EWs
    else:
        if np.random.binomial(1, A(Muv)):
            EW_cumsum = integrate.cumtrapz(1 / W(Muv) * np.exp(-Ws / W(Muv)),
                                           Ws)
            cumsum = EW_cumsum / EW_cumsum[-1]
            rn = np.random.uniform(size=1)
            EW_now = \
            np.interp(rn, np.concatenate((np.array([0.0]), cumsum)), Ws)[0]
            if return_lum:
                return EW_now, EW_now * C_const.value * L_UV_mean
            else:
                return EW_now
        else:
            if return_lum:
                return (0., 0.)
            else:
                return 0.


def tau_CGM(Muv):
    Muvs = np.load('/home/inikolic/projects/Lyalpha_bubbles/code/Lyman-alpha-bubbles/venv/data/Muv.npy')
    mh = np.load('/home/inikolic/projects/Lyalpha_bubbles/code/Lyman-alpha-bubbles/venv/data/mh.npy')
    mh_now = np.interp(Muv, np.flip(Muvs), np.flip(mh))
    v_c = ((10 * const.G * mh_now*u.M_sun * Cosmo.H(7.5))**( 1/3)).to(u.km/u.s).value
    if hasattr(Muv, '__len__'):
        tau_CGM = np.ones((len(Muv),100))
        for imi,mi in enumerate(Muv):
            for i_w,wv in enumerate(wave_em):
                if wave_to_dv(wv).value < v_c[imi]:
                    tau_CGM[imi,i_w] = 0.0

    else:
        tau_CGM = np.ones(100)
        for i_w, wv in enumerate(wave_em):
            if wave_to_dv(wv).value < v_c:
                tau_CGM[i_w] = 0.0
            else:
                break
    return tau_CGM


def calculate_number(
        fluct_level,
        x_dim=10,
        y_dim=10,
        z_dim=10,
        redshift=7.5,
        muv_cut=-19
):
    """
    Function calculates number of galaxies for a given setup which includes
    spatial dimensions, overdensity of the region (Lagrangian), redshift and
    muv cut.
    :param fluct_level: float,
        overdensity level of the region of interest.
    :param x_dim: float,
        x dimension of the region.
    :param y_dim: float,
        y dimension of the region.
    :param z_dim: float,
        z dimension of the region.
    :param redshift: float,
        redshift of the region
    :param muv_cut: float,
        Muv cut for galaxies.

    :return: N: int,
        number of galaxies to sample.
    """

    V = x_dim * y_dim * z_dim
    R_eq = np.sqrt(x_dim**2 + y_dim**2 + z_dim**2)

    Muvs = np.load(
        dir_all + 'venv/data/Muv.npy'
    )
    mh = np.load(
        dir_all + 'venv/data/mh.npy'
    )

    mh_cut = np.interp(muv_cut, np.flip(Muvs),np.flip( mh))

    hmf_this = chmf_func(z=redshift, delta_bias=0.0, R_bias=R_eq)
    hmf_this.prep_for_hmf_st(5.0, 15.0, 0.01)
    hmf_this.prep_collapsed_fractions(check_cache=False)

    masses = hmf_this.bins

    delta = hmf_this.sigma_cell * hmf_this.dicke * fluct_level

    mass_func = hmf_this.ST_hmf(delta)

    for index_to_stop, mass_func_element in enumerate(mass_func):
        if mass_func_element == 0 or np.isnan(mass_func_element):
            break
    masses = masses[:index_to_stop]
    mass_func = mass_func[:index_to_stop]

    N_cs = hmf_integral_gtm(masses, mass_func) * V
    #print(" these are cumulative numbers", N_cs)
    N = np.interp(np.log10(mh_cut), np.log10(masses[:len(N_cs)]), N_cs)
    #print("some numbers, mh_cut", mh_cut,"N in the end", N)
    return int(N)


def get_spectrum(
        cont,
        noise=None,
        resolution=None,
):
    """
    Function returns binned spectrum that mimics how an actual observation might
    look like.

    :param cont: ~numpy.array
        continuous spectrum
    :param noise: None or float,
        Noise level to be added to create the mock spectrum.
        If 'None', then no noise is added to the spectrum
    :param resolution: None or float,
        Spectral resolution of the mock power spectrum

    :return:
    bins: ~numpy.array:
        bins of the power spectrum.
    ps: ~numpy.array
        binned power spectrum.
    """
    bins = np.arange(wave_em[0].value, wave_em[-1].value, resolution)
    wave_em_dig = np.digitize(wave_em.value, bins)
    bins_po = np.append(bins, bins[-1] + spec_res)

    cont_flux = [
        np.trapz(
            x = wave_em.value[wave_em_dig == i+1],
            y = cont[wave_em_dig == i+1]
        )
        for i in range(len(bins))
    ]

    if noise is not None:
        cont_flux += np.random.normal(0, noise, len(bins))

    return 0.5 * (bins_po[1:] + bins_po[:-1]), cont_flux


def L_intr_AH22(Muv):
    def sigma(mh):
        if mh > 1e10:
            return 0.1  # log-normal
        else:
            return 1.0

    csv_AH_22 = np.loadtxt(
        '/home/inikolic/projects/Lyalpha_bubbles/Literature/AH+22/L_intr_Mass.csv',
        delimiter=','
    )
    mass = csv_AH_22[:, 0]
    L_intr = csv_AH_22[:, 1]
    Muvs = np.load(
        '/home/inikolic/projects/Lyalpha_bubbles/code/Lyman-alpha-bubbles/venv/data/Muv.npy')
    mh = np.load(
        '/home/inikolic/projects/Lyalpha_bubbles/code/Lyman-alpha-bubbles/venv/data/mh.npy')
    mh_now = np.interp(Muv, np.flip(Muvs), np.flip(mh))
    mean_L = np.interp(np.log10(mh_now), np.log10(mass), L_intr)

    print(np.log10(mh_now))
    if hasattr(mh_now, '__len__'):
        li = np.zeros(np.shape(Muv))
        for ind, (mi, meanil) in enumerate(
                zip(mh_now.flatten(), mean_L.flatten())):
            if mi > 1e10:
                li[ind] = 10 ** normal(meanil, 0.4)
            else:
                if binomial(0.5):
                    li[ind] = 10 ** normal(meanil, 1.0)
                else:
                    li[ind] = 0.0
        return li.reshape(np.shape(Muv))
    if mh_now > 1e10:
        return 10 ** normal(mean_L, 0.4)
    else:
        if binomial(0.5):
            return 10 ** normal(mean_L, 1.0)
        else:
            return 0.0
