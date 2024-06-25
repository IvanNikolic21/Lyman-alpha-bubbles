import numpy as np
from scipy import integrate
import os
import pandas as pd
from mpmath import nsum, exp, inf, log, ln
import scipy
from scipy.stats import gaussian_kde

from astropy import units as u
from astropy.cosmology import z_at_value
from astropy.cosmology import Planck18 as Cosmo
from astropy import constants as const

from venv.helpers import optical_depth, I, z_at_proper_distance

wave_em = np.linspace(1214, 1230., 100) * u.Angstrom
wave_Lya = 1215.67 * u.Angstrom
freq_Lya = (const.c / wave_Lya).to(u.Hz)
r_alpha = 6.25 * 1e8 / (4 * np.pi * freq_Lya.value)


def get_tl_data(
        datadir='/home/inikolic/projects/Lyalpha_bubbles/TL+23/',
        bg=True,
        z=8,
        xhi=0.8
):
    """
    Get Ting-Yi's data.

    :param datadir: string;
        where the data is stored
    :param bg: boolean;
        whether to use the rapid reionization model. Otherwise, fg is used.
    :param z: integer;
        Redshift to use. Options are 7,8,9
    :param xhi: float;
        Neutral fraction

    :return r_hist: numpy.array;
        Histogram values of radii.
    :return p_log_r_norm: numpy.array;
        histogram values of the distribution.
    """
    dir_bg_z8_r_all = datadir + 'bg_rall_z' + str(
        z) + '/'  # dir for full box bubble list [Rbub]
    dir_fg_z8_r_all = datadir + 'fg_rall_z' + str(z) + '/'

    if bg:
        box_dir = dir_bg_z8_r_all
    else:
        box_dir = dir_fg_z8_r_all

    bub_list = {}
    box_fills = np.sort(os.listdir(box_dir))
    for i, fil in enumerate(box_fills):
        ind = fil.find('nf')
        xhi = np.round(float(fil[ind + 2:ind + 6]), 1)
        data = pd.read_csv(box_dir + fil)
        bub_list[str(xhi)] = pd.DataFrame(
            data=np.vstack((data['R_cMpc'], data['R_cMpc'])).T,
            columns=['Muv', 'R_cMpc'])
    df = bub_list[str(xhi)]
    r_bubs = df.R_cMpc
    bins = np.logspace(np.log10(0.4), np.log10(1.1 * np.max(r_bubs)), 12)
    pr_hist, r_hist_edge = np.histogram(r_bubs, density=True, bins=bins)
    r_hist = 0.5 * (r_hist_edge[:-1] + r_hist_edge[1:])  # calculate the centers
    yy = pr_hist * r_hist  # calculate the y values for the dP/dlogR plot

    p_log_r_norm = yy / np.trapz(yy, x=r_hist)  # normalize the distribution
    return r_hist, p_log_r_norm


def bubble_size_distro(
        r,
        r_hist=None,
        p_log_r_norm=None,
        redshift_fit=False,
        xH=None,
):
    """
    Calculate the PDF of the bubble size distribution.
    Inputs
    ----------
    r: float
        Bubble size for which we evaluate the distribution.
    r_hist: None or list-like
        If given, it contains radii-bins for the arbitrary distribution.
    p_log_r_norm: None or list-like
        If given, it contains pdf-values for radii-bins of the distribution.

    Output
    ----------
    PDF_R: float,
        Value of the PDF at a given R.

    Note:
    The function evaluates the given PDF if its values are given. Else, it uses
    a mock distribution used in some plots.
    """

    if type(r_hist) != type(p_log_r_norm) and len(r_hist) != len(p_log_r_norm):
        raise ValueError("Your manual PDF arrays are inconsistent")

    if redshift_fit:
        if xH is None:
            raise ValueError("You need to specify neutral fraction for redshift fit")
        sig_par = np.array([ 211.80041642, -420.19373628,  211.92984014])
        A_par = np.array([ 29.62577517, -51.82492007,  31.80019508,  -5.99811482])
        tau_par = np.array([-137.83789264,  333.40850312, -273.79098176,   77.66713409])

        sig = sig_par[0] * xH**2 + sig_par[1] * xH + sig_par[2]
        A = A_par[0] * xH**3 + A_par[1] * xH**2 + A_par[2] * xH + A_par[3]
        tau = tau_par[0] * xH**3 + tau_par[1] * xH**2 + tau_par[2] * xH + tau_par[3]

        def f_bub(x, sig_in, A_in, tau_in):
            return A_in / 2 / np.pi / np.sqrt(sig_in) * np.exp(
                -0.5 * (x - 0.0) ** 2 / sig ** 2) * scipy.special.erf(
                (x - 0.0) / tau_in)

        return f_bub(r, sig, A, tau)


    if r_hist is None:
        r = np.log10(r)
        mu = np.log10(5.0)
        sig = 0.5
        return 1 / np.sqrt(2 * np.pi) / sig * np.exp(-(r - mu) ** 2 / sig ** 2)

    else:
        return np.interp(r, r_hist, p_log_r_norm)


def get_bubbles(
        xh,
        z_v,
        use_tl_result=True,
        mock=False,
):
    """
    Function takes in some bubble distribution and returns an instance of
    bubbles.

    Input:_
    ----------
    xh: float,
        neutral fraction at which we're calculating the distribution.
    z_v: float,
        the extent in Mpc of bubbles away from the main one.
    use_tl_result: boolean,
        whether to use TL+23 bubble size distribution.

    Returns:
    bubble_xs: list of floats,
        x-positions of bubbles found.
    bubble_ys: list of floats,
        y-positions of bubbles found.
    bubble_zs: list of floats,
        z-positions of bubbles found. This is a LoS axis
    bubble_rs: list of floats,
        radii of bubbles found.
    ---------

    Note:
    The code takes a fixed bubble_size_distro function which is not optimal if
    I want to change it. Something to think about.

    """
    bubble_xs = []
    bubble_ys = []
    bubble_zs = []
    bubble_rs = []
    v_tot = 0.0
    r_min = np.log10(1.0)
    r_max = 40.0
    log_r_max = np.log10(40.0)
    if use_tl_result:
        # r_hist, p_log_r_norm = get_tl_data(
        #     xhi=xh
        # )
        rs = np.logspace(r_min, log_r_max, 1000)
        cdf = integrate.cumtrapz(
            bubble_size_distro(
                rs,
                r_hist=None,
                p_log_r_norm=None,  # better way to do this tomorrow
                redshift_fit=False,
                xH=xh,
            ) / rs,
            rs
        )
        cdf = cdf.flatten()
    else:
        rs = np.logspace(-1, 3, 1000)
        cdf = integrate.cumtrapz(
            bubble_size_distr(
                rs
            ),
            rs
        )
    vol_box = z_v * r_max * r_max
    try_i = 0
    tolerance = 0.01
    that_it = False
    cdf_norm = cdf/cdf[-1]

    while abs(v_tot - (1 - xh) * 3.2* vol_box) / (
            (1 - xh) * 3.2*vol_box) > tolerance:

        random_numb = np.random.uniform(size=1)
        bubble_now = np.interp(random_numb, cdf_norm, rs[:-1])

        random_x = np.random.uniform(-r_max, r_max)
        random_y = np.random.uniform(-r_max, r_max)
        random_z = np.random.uniform(0, z_v)
        if mock and (bubble_now+10.0)**2 > random_x**2 + random_y**2 + random_z**2:
                continue #for the mock we don't want small bubbles to intersect
                        # the big one.

        v = 4. * np.pi / 3. * bubble_now ** 3
        v_ded = 0.0

        if random_x < -r_max + bubble_now:
            h = -r_max - random_x + bubble_now
            v_ded = np.pi * 2. / 3. * h ** 2 * (3 * bubble_now - h)
        elif random_x > r_max - bubble_now:
            h = bubble_now - (r_max - random_x)
            v_ded = np.pi * 2. / 3. * h ** 2 * (3 * bubble_now - h)

        v = v - v_ded
        v_ded = 0.
        if random_y < -r_max + bubble_now:
            h = -r_max - random_y + bubble_now
            v_ded = np.pi * 2. / 3. * h ** 2 * (3 * bubble_now - h)
        elif random_y > r_max - bubble_now:
            h = bubble_now - (r_max - random_y)
            v_ded = np.pi * 2. / 3. * h ** 2 * (3 * bubble_now - h)

        v = v - v_ded
        v_ded = 0.
        if random_z < 0 + bubble_now:
            h = bubble_now - random_z
            v_ded = np.pi * 2. / 3. * h ** 2 * (3 * bubble_now - h)
        elif random_z > z_v - bubble_now:
            h = bubble_now - (z_v - random_z)
            v_ded = np.pi * 2. / 3. * h ** 2 * (3 * bubble_now - h)

        v = v - v_ded
        break_it = False
        v_ded = 0
        for bx, by, bz, br in zip(bubble_xs, bubble_ys, bubble_zs, bubble_rs):
            dist_mod = (bx - random_x) ** 2 + (by - random_y) ** 2 + (
                    bz - random_z) ** 2
            if dist_mod < (br + bubble_now) ** 2:
                dist = np.sqrt(dist_mod)
                if br > bubble_now:
                    big = br
                    small = bubble_now
                else:
                    small = br
                    big = bubble_now
                if dist + small < big:
                    break_it = True
                v_ded += (np.pi * (big + small - dist) ** 2 * (
                        dist ** 2 + 2 * dist * small - 3 * small ** 2 + 2 *
                        dist * big + 6 * big * small - 3 * big ** 2)
                          / 12 / dist
                          )
        v = v - v_ded
        if break_it or v < 0:

            try_i += 1
            if try_i >= 5:
                break
            else:
                continue

        ion_vol = (1 - xh) * 3.2 * vol_box
        vtotpv = v_tot + v
        if vtotpv / (ion_vol) > 1.0:
            if abs(vtotpv - ion_vol) / vtotpv < tolerance:
                that_it = True
            else:
                continue

        v_tot += v
        try_i = 0
        bubble_xs.append(random_x)
        bubble_ys.append(random_y)
        bubble_zs.append(random_z)
        bubble_rs.append(bubble_now)
        if that_it:
            break
    return bubble_xs, bubble_ys, bubble_zs, bubble_rs


def calculate_taus(
        x_small,
        y_small,
        z_small,
        r_bubbles,
        red_s,
        z_end_bub,
        n_iter=1000,
        x_pos = None,
        y_pos = None,
):
    """
    Calculates taus for random sightlines for a given bubble configuration.

    :param x_small: numpy.array;
        x-coordinates of centers of outside bubbles.
    :param y_small: numpy.array;
        y-coordinates of centers of outside bubbles.
    :param z_small: numpy.array;
        z-coordinates of centers of outside bubbles.
    :param r_bubbles: numpy.array;
        radii of outside bubbles.
    :param red_s: float;
        redshift of the center of the main bubble.
    :param z_end_bub: float;
        redshift of the edge from which we calculate taus.
    :param n_iter: integer;
        number of random sightlines sampled.
    :param x_pos: float,
        x-position that is taken.
    :param y_pos: float,
        y-position that is taken.

    :return taus: numpy.arrayM
        Optical depths for a given configuration.

    Note: If position is specified, this is the only LoS to be calculated.
    """
    if x_pos is not None and y_pos is not None:
        x_random = [x_pos]
        y_random = [y_pos]
    else:
        x_random = np.random.uniform(-10, 10, size=n_iter)
        y_random = np.random.uniform(-10, 10, size=n_iter)

    taus = []

    z_start = z_end_bub

    for i, (xr, yr) in enumerate(zip(x_random, y_random)):
        z_edge_up = []
        z_edge_lo = []
        red_edge_up = []
        red_edge_lo = []
        for xb, yb, zb, rb in zip(x_small, y_small, z_small, r_bubbles):
            # initialize lists of intersections.
            # check if galaxy will intersect the bubble.
            # note that bubbles behind the galaxy don't influence it.
            if (xr - xb) ** 2 + (yr - yb) ** 2 < rb ** 2:
                # calculate z-s and redshifts at which it'll happen.
                z_edge_up_i = zb - np.sqrt(
                    rb ** 2 - ((xr - xb) ** 2 + (yr - yb) ** 2))
                z_edge_lo_i = zb + np.sqrt(
                    rb ** 2 - ((xr - xb) ** 2 + (yr - yb) ** 2))
                # get the redshifts
                red_edge_up_i = z_at_value(
                    Cosmo.comoving_distance,
                    Cosmo.comoving_distance(
                        7.5) - z_edge_up_i * u.Mpc - 10 * u.Mpc
                    # the radius of the big bubble
                )
                red_edge_lo_i = z_at_value(
                    Cosmo.comoving_distance,
                    Cosmo.comoving_distance(
                        7.5) - z_edge_lo_i * u.Mpc - 10 * u.Mpc
                    # the radius of the big bubble
                )
                z_edge_up.append(np.copy(z_edge_up_i)[0])
                z_edge_lo.append(np.copy(z_edge_lo_i)[0])
                red_edge_up.append(np.copy(red_edge_up_i)[0])
                red_edge_lo.append(np.copy(red_edge_lo_i)[0])
        z_edge_up = np.array(z_edge_up)
        z_edge_lo = np.array(z_edge_lo)
        red_edge_up = np.array(red_edge_up)
        red_edge_lo = np.array(red_edge_lo)
        if len(z_edge_up) == 0:
            # galaxy doesn't intersect any of the bubbles:
            taus.append(
                np.array(optical_depth(
                    wave_em,
                    t=1 * u.K,
                    z_min=5.5,
                    z_max=z_start,
                    z_s=7.5,
                    z_bubble_center=7.5,
                    inside_hii=False
                )
                )
            )
            continue
        indices_up = z_edge_up.argsort()

        z_edge_up_sorted = z_edge_up[indices_up]
        z_edge_lo_sorted = z_edge_lo[indices_up]
        red_edge_up_sorted = red_edge_up[indices_up]
        red_edge_lo_sorted = red_edge_lo[indices_up]

        indices_to_del_lo = []
        indices_to_del_up = []

        for j in range(len(z_edge_up) - 1):
            if len(z_edge_up_sorted) != 1:
                if red_edge_lo_sorted[j] < red_edge_up_sorted[j + 1]:
                    # got an overlapping bubble
                    indices_to_del_lo.append(j)
                    indices_to_del_up.append(j + 1)
        z_edge_lo_sorted = np.delete(z_edge_lo_sorted, indices_to_del_lo)
        z_edge_up_sorted = np.delete(z_edge_up_sorted, indices_to_del_up)
        red_edge_up_sorted = np.delete(red_edge_up_sorted, indices_to_del_up)
        red_edge_lo_sorted = np.delete(red_edge_lo_sorted, indices_to_del_lo)
        tau_i = np.zeros(len(wave_em))

        len_z = len(z_edge_up_sorted)

        for index, (z_up_i, z_lo_i, red_up_i, red_lo_i) in enumerate(
                zip(z_edge_up_sorted, z_edge_lo_sorted, red_edge_up_sorted,
                    red_edge_lo_sorted)):
            if index == 0:
                if z_up_i < 0 and z_lo_i < 0:
                    print("wrong bubble, it's above the galaxy.")
                    raise ValueError
                if z_up_i < 0 < z_lo_i:
                    tau_i = np.array(
                        optical_depth(
                            wave_em,
                            t=1e4 * u.K,
                            z_min=red_lo_i,
                            z_max=z_start,
                            z_s=red_s,
                            z_bubble_center=red_s,
                            inside_hii=True
                        )
                    )
                else:

                    tau_i = np.array(
                        optical_depth(
                            wave_em,
                            t=1 * u.K,
                            z_min=red_up_i,
                            z_max=z_start,
                            z_s=red_s,
                            z_bubble_center=7.5,
                            inside_hii=False
                        )
                    )
                    tau_i += np.array(
                        optical_depth(
                            wave_em,
                            t=1e4 * u.K,
                            z_min=red_lo_i,
                            z_max=red_up_i,
                            z_s=red_s,
                            z_bubble_center=7.5,
                            inside_hii=True
                        )
                    )
            elif index != len(z_edge_up_sorted) - 1:
                tau_i += np.array(
                    optical_depth(
                        wave_em,
                        t=1 * u.K,
                        z_min=red_up_i,
                        z_max=red_edge_lo_sorted[index - 1],
                        z_s=red_s,
                        z_bubble_center=7.5,
                        inside_hii=False
                    )
                )
                tau_i += np.array(
                    optical_depth(
                        wave_em,
                        t=1e4 * u.K,
                        z_min=red_lo_i,
                        z_max=red_up_i,
                        z_s=red_s,
                        z_bubble_center=7.5,
                        inside_hii=True
                    )
                )
            if index == len_z - 1 and not len_z == 1:
                tau_i += np.array(
                    optical_depth(
                        wave_em,
                        t=1 * u.K,
                        z_min=red_up_i,
                        z_max=red_edge_lo_sorted[index - 1],
                        z_s=red_s,
                        z_bubble_center=7.5,
                        inside_hii=False
                    )
                )
                tau_i += np.array(
                    optical_depth(
                        wave_em,
                        t=1e4 * u.K,
                        z_min=red_lo_i,
                        z_max=red_up_i,
                        z_s=red_s,
                        z_bubble_center=7.5,
                        inside_hii=True
                    )
                )
                tau_i += np.array(
                    optical_depth(
                        wave_em,
                        t=1 * u.K,
                        z_min=5.5,
                        z_max=red_lo_i,
                        z_s=red_s,
                        z_bubble_center=7.5,
                        inside_hii=False
                    )
                )
            elif index == len_z - 1 and len_z == 1:
                tau_i += np.array(
                    optical_depth(
                        wave_em,
                        t=1 * u.K,
                        z_min=5.5,
                        z_max=red_lo_i,
                        z_s=red_s,
                        z_bubble_center=7.5,
                        inside_hii=False
                    )
                )
        taus.append(tau_i)
    return taus


def calculate_taus_i(
        x_small,
        y_small,
        z_small,
        r_bubbles,
        z_source,
        z_end_bubble,
        n_iter=500,
        x_pos=None,
        y_pos=None,
        dist = 10,
):
    """
        A different option to calculate tau distribution.
        Calculate a couple of taus of a galaxy that is located in zs.
        If x_pos and y_pos are given, then they determine the sightline used.
    """
    x_small = np.array(x_small)
    y_small = np.array(y_small)
    z_small = np.array(z_small)
    r_bubbles = np.array(r_bubbles).flatten()

    if x_pos is not None and y_pos is not None:
        x_random = [x_pos]
        y_random = [y_pos]
    else:
        x_random = np.random.uniform(-40, 40, size=n_iter)
        y_random = np.random.uniform(-40, 40, size=n_iter)
    taus = np.zeros((n_iter, len(wave_em)))
    z = wave_em.value / 1215.67 * (1 + z_source) - 1
    one_over_onepz = 1/(1+z)
    one_over_onesource = 1/(1+z_source)

    tau_gp = 7.16 * 1e5 * ((1 + z_source) / 10) ** 1.5
    tau_pref = tau_gp * r_alpha / np.pi
    for i, (xr, yr) in enumerate(zip(x_random, y_random)):
        dist_arr = np.sqrt(
            r_bubbles ** 2 - ((xr - x_small) ** 2 + (yr - y_small) ** 2)
        )

        # t0  =time.time()
        z_edge_up_arr = z_small - dist_arr

        z_edge_lo_arr = z_small + dist_arr

        red_edge_up_arr = z_at_proper_distance(
            (z_edge_up_arr + 10) * one_over_onesource * u.Mpc, 7.5
            # needs to be fixed
        )
        red_edge_lo_arr = z_at_proper_distance(
            (z_edge_lo_arr + 10) * one_over_onesource * u.Mpc, 7.5
        )
        z_edge_up = z_edge_up_arr[~ np.isnan(z_edge_up_arr)]
        z_edge_lo = z_edge_lo_arr[~ np.isnan(z_edge_lo_arr)]
        red_edge_up = red_edge_up_arr[~ np.isnan(red_edge_up_arr)]
        red_edge_lo = red_edge_lo_arr[~ np.isnan(red_edge_lo_arr)]

        #|| up is the shorter version of the code that was there once searching
        #where small bubbles intersect each sighltine

        if len(z_edge_up) == 0:
            # galaxy doesn't intersect any of the bubbles:
            taus[i,:] = tau_wv(
                wave_em,
                dist=dist,
                zs=z_source,
                z_end=5.3,
                nf=0.8
            )
            # to be checked
            continue
        indices_up = z_edge_up.argsort()

        z_edge_up_sorted = z_edge_up[np.flip(indices_up)]
        z_edge_lo_sorted = z_edge_lo[np.flip(indices_up)]
        red_edge_up_sorted = red_edge_up[np.flip(indices_up)]
        red_edge_lo_sorted = red_edge_lo[np.flip(indices_up)]

        indices_to_del_lo = []
        indices_to_del_up = []
        for i_fi in range(len(z_edge_up) - 1):
            if len(z_edge_up_sorted) != 1:
                if z_edge_lo_sorted[i_fi] < z_edge_up_sorted[i_fi + 1]:
                    # got an overlapping bubble
                    indices_to_del_lo.append(i_fi)
                    indices_to_del_up.append(i_fi + 1)
        z_edge_lo_sorted = np.delete(z_edge_lo_sorted, indices_to_del_lo)
        z_edge_up_sorted = np.delete(z_edge_up_sorted, indices_to_del_up)
        red_edge_up_sorted = np.delete(red_edge_up_sorted, indices_to_del_up)
        red_edge_lo_sorted = np.delete(red_edge_lo_sorted, indices_to_del_lo)
        tau_i = np.zeros(len(wave_em))
        for index, (z_up_i, z_lo_i, red_up_i, red_lo_i) in enumerate(
                zip(z_edge_up_sorted, z_edge_lo_sorted, red_edge_up_sorted,
                    red_edge_lo_sorted)):
            if index == 0:
                if z_up_i < 0 and z_lo_i < 0:
                    print("wrong bubble, it's above the galaxy.")
                    raise ValueError
                if z_up_i < 0 < z_lo_i:
                    z_bi = z_end_bubble
                    z_ei = red_lo_i
                    zb_ar = (1 + z_bi) * one_over_onepz
                    tau_i = tau_pref * zb_ar ** 1.5 * (
                            I(zb_ar) - I((1 + z_ei) * one_over_onepz)
                    )
                else:
                    z_bi = z_end_bubble
                    z_ei = red_up_i
                    zb_ar = (1 + z_bi) * one_over_onepz
                    tau_i = tau_pref * zb_ar ** 1.5 * (
                            I(zb_ar) - I((1 + z_ei) * one_over_onepz)
                    )
            elif index != len(z_edge_up_sorted) - 1:
                z_bi = red_edge_lo_sorted[index - 1]
                z_ei = red_up_i
                zb_ar = (1 + z_bi) * one_over_onepz
                tau_i += tau_pref * zb_ar ** 1.5 * (
                        I(zb_ar) - I((1 + z_ei) * one_over_onepz)
                )
            if index == len(z_edge_up_sorted)-1 and len(z_edge_up_sorted) != 1:
                z_bi = red_edge_lo_sorted[index - 1]
                z_ei = red_up_i
                zb_ar = (1 + z_bi) * one_over_onepz
                tau_i += tau_pref * zb_ar ** 1.5 * (
                        I(zb_ar) - I((1 + z_ei) * one_over_onepz)
                )
                z_bi = red_lo_i
                z_ei = 5.3

                zb_ar = (1 + z_bi) * one_over_onepz
                tau_i += tau_pref * zb_ar ** 1.5 * (
                        I(zb_ar) - I((1 + z_ei) * one_over_onepz)
                )
            elif index == len(z_edge_up_sorted) - 1 and len(
                    z_edge_up_sorted) == 1:
                z_bi = red_lo_i
                z_ei = 5.3
                zb_ar = (1 + z_bi) * one_over_onepz
                tau_i += tau_pref * zb_ar ** 1.5 * (
                        I(zb_ar) - I((1 + z_ei) * one_over_onepz
                                     )
                )
        try:
            taus[i,:]= tau_i
        except IndexError:
            if n_iter==1:
                taus = tau_i
            else:
                raise IndexError("Something else")
    taus = taus.flatten()
    taus[taus<0.0]=np.inf

    return taus.reshape((n_iter, len(wave_em)))


def tau_wv(
        wv,
        dist=10,
        zs=7.5,
        z_end=5.3,
        nf=0.5
):
    z_b1 = z_at_value(
        Cosmo.comoving_distance,
        Cosmo.comoving_distance(zs) - dist*u.Mpc
    )
    z = wv.value/1216 * (1+zs) - 1
    tau_gp = 7.16 * 1e5 * ((1+zs)/10)**1.5
    r_alpha = 6.25 * 1e8 / (4 * np.pi * freq_Lya.value)

    def func(x, f=nf):
        f = f/(1-f)
        return 0.5 / x / f * (ln(
            (2*x-1)/(2*x+1)
        ) + ln(
            (4*x*f + 2 * x + 1) / (4 * x*f + 2 * x-1)
        )
        )

    # x_d = 1 + 0.5 * float(nsum(func, [1, inf],))
    x_d = nf  # actually, nf might be better for what we're doing
    tau = tau_gp * r_alpha / np.pi * x_d * (
            (1+z_b1)/(1+z)
    )**1.5 * (
            I((1+z_b1)/(1+z)) - I((1+z_end)/(1+z))
    )
    
    return tau


def get_xH(
        z,
        main_dir="home/inikolic/projects/Lyalpha_bubbles/code/Lyman-alpha-bubbles/",
):
    """
        Function returns neutral fraction as a sample from the distribution at a
        given redshift

    :param z: float
        redshift at which we get the sample.
    :return:
    """

    xHs_all = np.load(main_dir + '/venv/data/global_xHs.npy')
    node_r = np.load(main_dir + '/venv/data/node_r.npy')
    xHs = []
    for x in xHs_all:
        xHs.append(np.interp(z, np.flip(node_r), np.flip(x)))

    gk = gaussian_kde(xHs)
    return gk.resample(1)
