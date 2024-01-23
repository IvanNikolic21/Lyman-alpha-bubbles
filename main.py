import numpy as np
from numpy.linalg import LinAlgError
import argparse
import os
from scipy import integrate
from scipy.stats import gaussian_kde

from astropy.cosmology import z_at_value
from astropy import units as u
from astropy.cosmology import Planck18 as Cosmo

import itertools
from joblib import Parallel, delayed

from venv.galaxy_prop import get_js, get_mock_data, calculate_EW_factor, p_EW
from venv.galaxy_prop import get_muv, tau_CGM, calculate_number
from venv.igm_prop import get_bubbles
from venv.igm_prop import calculate_taus_i, get_xH

from venv.helpers import z_at_proper_distance

wave_em = np.linspace(1213, 1221., 100) * u.Angstrom
wave_Lya = 1215.67 * u.Angstrom


def _get_likelihood(
        ndex,
        xb,
        yb,
        zb,
        rb,
        xs,
        ys,
        zs,
        tau_data,
        n_iter_bub,
        redshift=7.5,
        muv=None,
        include_muv_unc=False,
        beta_data=None,
        use_EW=False,
        xH_unc=False,
        la_e=None,
        flux_limit=1e-18,
        like_on_flux=False,
):
    """

    :param ndex: integer;
        index of the calculation
    :param xb: float;
        x-coordinate of the bubble
    :param yb: float;
        y-coordinate of the bubble
    :param zb: float;
        z-coordinate of the bubble
    :param rb: float;
        Radius of the bubble.
    :param xs: list;
        x-coordinates of galaxies;
    :param ys: list;
        y-coordinates of galaxies;
    :param zs: list;
        z-coordinates of galaxies;
    :param tau_data: list;
        transmissivities of galaxies.
    :param n_iter_bub: integer;
        how many times to iterate over the bubbles.
    :param include_muv_unc: boolean;
        whether to include the uncertainty in the Muv.
    :param beta_data: numpy.array or None.
        UV-slopes for each of the mock galaxies. If None, then a default choice
        of -2.0 is used.
    :param use_EW: boolean
        Whether likelihood calculation is done on flux and other flux-related
        quantities, unlike the False option when the likelihood is calculated
        on transmission directly.

    :return:

    Note: to be used only in the sampler
    """

    if like_on_flux is not False:
        spec_res = wave_Lya.value * (1 + redshift) / 2700
        bins = np.arange(wave_em.value[0] * (1 + redshift),
                         wave_em.value[-1] * (1 + redshift), spec_res)
        wave_em_dig = np.digitize(wave_em.value * (1 + redshift), bins)
        bins_po = np.append(bins, bins[-1] + spec_res)

    likelihood = 0.0
    taus_tot = []
    flux_tot = []
    spectrum_tot = []
    if la_e is not None:
        flux_mock = np.zeros(len(xs))

    # For these parameters let's iterate over galaxies
    if beta_data is None:
        beta_data = np.zeros(len(xs))
    reds_of_galaxies = np.zeros(len(xs))
    for index_gal, (xg, yg, zg, muvi, beti, li) in enumerate(
            zip(xs, ys, zs, muv, beta_data, la_e)
    ):
        taus_now = []
        flux_now = []
        if like_on_flux is not False:
            spectrum_now = []
        # red_s = z_at_value(
        #     Cosmo.comoving_distance,
        #     Cosmo.comoving_distance(redshift) + zg * u.Mpc,
        #     ztol=0.00005
        # )
        red_s = z_at_proper_distance(
            - zg / (1 + redshift) * u.Mpc, redshift
        )
        reds_of_galaxies[index_gal] = red_s
        # calculating fluxes if they are given
        if la_e is not None:
            flux_mock[index_gal] = li / (
                    4 * np.pi * Cosmo.luminosity_distance(
                red_s).to(u.cm).value**2
            )

        if ((xg - xb) ** 2 + (yg - yb) ** 2
                + (zg - zb) ** 2 < rb ** 2):
            dist = zg - zb + np.sqrt(
                rb ** 2 - (xg - xb) ** 2 - (yg - yb) ** 2
            )
            # z_end_bub = z_at_value(
            #     Cosmo.comoving_distance,
            #     Cosmo.comoving_distance(red_s) - dist * u.Mpc,
            #     ztol=0.00005
            # )
            z_end_bub = z_at_proper_distance( dist / (1+ redshift) * u.Mpc, redshift)
        else:
            z_end_bub = red_s
            dist = 0
        for n in range(n_iter_bub):
            j_s = get_js(muv=muvi, n_iter=50, include_muv_unc=include_muv_unc)
            if xH_unc:
                x_H = get_xH(redshift)  # using the central redshift.
            else:
                x_H=0.65
            x_outs, y_outs, z_outs, r_bubs = get_bubbles(
                x_H,
                300
            )
            tau_now_i = calculate_taus_i(
                x_outs,
                y_outs,
                z_outs,
                r_bubs,
                red_s,
                z_end_bub,
                n_iter=50,
                dist=dist,
            )
            eit_l = np.exp(-np.array(tau_now_i))
            tau_cgm_gal = tau_CGM(muvi)
            res = np.trapz(
                eit_l * tau_cgm_gal * j_s[0] / integrate.trapz(
                    j_s[0][0],
                    wave_em.value),
                wave_em.value
            )
            #EW_data = calculate_EW_factor(muvi, beti) * np.array(tau_data_I)

            if np.all(np.array(res) < 10000):
                if use_EW:
                    taus_now.extend(
                        res.tolist()
                    )
                else:
                    taus_now.extend(np.array(res).tolist()) # just to be sure
            else:
                print("smth wrong", res, flush=True )

            lae_now = np.array(
                [p_EW(muvi, beti, )[1] for blah in range(len(taus_now))]
            )
            flux_now = lae_now * np.array(taus_now).flatten() / (
                            4 * np.pi * Cosmo.luminosity_distance(
                        red_s).to(u.cm).value**2
            )
            #print(np.shape(lae_now[49] * j_s[0][49] *np.exp(-tau_now_i[49])* tau_CGM(muvi)), len(taus_now),flush=True)
            spectrum_now = np.array([[
                    np.trapz(x=wave_em.value[wave_em_dig == i + 1],
                             y=(lae_now[ind_igor] * j_s[0][ind_igor] * np.exp(
                                 -tau_now_i[ind_igor]
                            ) * tau_CGM(muvi) / (
                                            4 * np.pi * Cosmo.luminosity_distance(
                                    red_s).to(u.cm).value ** 2) / integrate.trapz(
                              j_s[0][ind_igor],
                                wave_em.value)
                                )[wave_em_dig == i + 1]) for i in range(len(bins))
                ] for ind_igor in range(len(eit_l))])
            spectrum_now = np.array(spectrum_now)
            spectrum_now += np.random.normal(
                0,
                1e-19,
                np.shape(spectrum_now)
            )
        flux_tot.append(np.array(flux_now).flatten())
        taus_tot.append(np.array(taus_now).flatten())
        spectrum_tot.append(spectrum_now)

    taus_tot_b = []
    #print(np.shape(taus_tot_b), np.shape(tau_data), flush=True)

    try:
        taus_tot_b = []
        flux_tot_b = []
        spectrum_tot_b = []
        for fi,li, speci in zip(flux_tot,taus_tot, spectrum_tot):
            if np.all(np.array(li) < 10000.0): # maybe unnecessary
                taus_tot_b.append(li)
                flux_tot_b.append(fi)
                spectrum_tot_b.append(speci)
#        print(np.shape(taus_tot_b), np.shape(tau_data), flush=True)

        for ind_data, (flux_line,tau_line, spec_line) in enumerate(
                zip(np.array(flux_tot_b),np.array(taus_tot_b),np.array(spectrum_tot_b))
        ):
            tau_kde = gaussian_kde((np.array(tau_line)))
            flux_kde = gaussian_kde((np.array(flux_line)))
            if like_on_flux is not False:
                spec_kde = [gaussian_kde((np.array(spec_line)[:,i_b])) for i_b in range(len(bins))]
            if la_e is not None:
                flux_tau = flux_mock[ind_data] * tau_data[ind_data]

            if not use_EW:
                if tau_data[ind_data] < 3:
                    likelihood += np.log(tau_kde.integrate_box(0, 3))
                else:
                    likelihood += np.log(tau_kde.evaluate((tau_data[ind_data])))

            else:
                if like_on_flux is not False:
                    for bi in range(len(bins)):
                        if like_on_flux[ind_data,bi] < 1e-19:
                            likelihood += np.log(spec_kde[bi].integrate_box(0, 1e-19))
                        else:
                            likelihood += np.log(spec_kde[bi].evaluate(like_on_flux[ind_data,bi]))
                else:
                    if flux_tau < flux_limit:
                        # _, la_limit_muv = p_EW(
                        #     muv[ind_data],
                        #     beta_data[ind_data],
                        #     mean=True,
                        #     return_lum=True,
                        # )
                        # flux_limit_muv = la_limit_muv / (
                        #         4 * np.pi * Cosmo.luminosity_distance(
                        #     red_s).to(u.cm).value**2
                        # ) #TODO implement redshift dependence
                        # tau_limit = flux_limit/flux_limit_muv
                        print("This galaxy failed the tau test, it's flux is", flux_tau)
                        likelihood += np.log(flux_kde.integrate_box(0, flux_limit))
                        print("It's integrate likelihood is", flux_kde.integrate_box(0, flux_limit))
                    else:
                        print("all good", flux_tau)
                        likelihood += np.log(flux_kde.evaluate(flux_tau))
        # print(
        #     np.array(taus_tot),
        #     np.array(tau_data),
        #     np.shape(taus_tot),
        #     np.shape(tau_data),
        #     tau_kde.evaluate(tau_data),
        #     "This is what evaluate does for this params",
        #     xb, yb, zb, rb  , flush=True
        # )
    except (LinAlgError, ValueError, TypeError):
        likelihood += -np.inf
        print("OOps there was valu error, let's see why:", flush=True)
        print(tau_data, flush=True)
        print(taus_tot_b, flush=True)
        raise TypeError
    if hasattr(likelihood, '__len__'):
        return ndex, np.product(likelihood)
    else:
        return ndex, likelihood


def sample_bubbles_grid(
        tau_data,
        xs,
        ys,
        zs,
        n_iter_bub=100,
        n_grid=10,
        redshift=7.5,
        muv=None,
        include_muv_unc=False,
        beta_data=None,
        use_EW=False,
        xH_unc=False,
        la_e=None,
        multiple_iter=False,
        flux_limit=1e-18,
        like_on_flux=False,
):
    """
    The function returns the grid of likelihood values for given input
    parameters.

    :param tau_data: list of floats;
        List of transmission that represent our data.
    :param xs: list of floats;
        List of x-positions that represent our data.
    :param ys: list of floats,
        List of y-positions that represent our data.
    :param zs: list of floats,
        List of z-positions that represent our data. LoS axis.
    :param n_iter_bub: integer,
        number of outside bubble configurations per iteration.
    :param n_grid: integer,
        number of grid points.
    :param muv: muv,
        UV magnitude data
    :param include_muv_unc: boolean,
        whether to include muv uncertainty.
    :param beta_data: float,
        beta data.
    :param use_EW: boolean
        whether to use EW or transmissions directly.
    :param xH_unc: boolean
        whether to use uncertainty in the underlying neutral fraction in the
        likelihood analysis
    :param la_e: ~np.array or None
        If provided, it's a numpy array consisting of Lyman-alpha total
        luminosity for each mock-galaxy.

    :return likelihood_grid: np.array of shape (N_grid, N_grid, N_grid, N_grid);
        likelihoods for the data on a grid defined above.
    """

    # first specify a range for bubble size and bubble position
    r_min = 5  # small bubble
    #r_max = 37  # bubble not bigger than the actual size of the box
    r_max = 30
    r_grid = np.linspace(r_min, r_max, n_grid)

    x_min = -15.5
    x_max = 15.5
    x_grid = np.linspace(x_min, x_max, n_grid)

    y_min = -15.5
    y_max = 15.5
    y_grid = np.linspace(y_min, y_max, n_grid)

    #z_min = -12.5
    #z_max = 12.5
    z_min = -5.0
    z_max = 5.0
    r_min = 5.0
    r_max = 15.0
    z_grid = np.linspace(z_min, z_max, n_grid)
    #x_grid = np.linspace(x_min, x_max, n_grid)[5:6]
    #y_grid = np.linspace(y_min, y_max, n_grid)[5:6]
    x_grid = np.linspace(0.0,0.0,1)
    y_grid = np.linspace(0.0,0.0,1)
    r_grid = np.linspace(r_min,r_max,n_grid)
    if multiple_iter:
        like_grid_top = np.zeros(
            (len(x_grid), len(y_grid), len(z_grid), len(r_grid), multiple_iter)
        )

        for ind_iter in range(multiple_iter):
            if like_on_flux is not False:
                like_on_flux = like_on_flux[ind_iter]
            like_calc = Parallel(
                n_jobs=50
            )(
                delayed(
                    _get_likelihood
                )(
                    index,
                    xb,
                    yb,
                    zb,
                    rb,
                    xs[ind_iter],
                    ys[ind_iter],
                    zs[ind_iter],
                    tau_data[ind_iter],
                    n_iter_bub,
                    redshift=redshift,
                    muv=muv[ind_iter],
                    include_muv_unc=include_muv_unc,
                    beta_data=beta_data[ind_iter],
                    use_EW=use_EW,
                    xH_unc=xH_unc,
                    la_e=la_e[ind_iter],
                    flux_limit=flux_limit,
                    like_on_flux=like_on_flux,
                ) for index, (xb, yb, zb, rb) in enumerate(
                    itertools.product(x_grid, y_grid, z_grid, r_grid)
                )
            )
            like_calc.sort(key=lambda x: x[0])
            likelihood_grid = np.array([l[1] for l in like_calc])
            likelihood_grid = likelihood_grid.reshape(
                (len(x_grid), len(y_grid), len(z_grid), len(r_grid))
            )
            like_grid_top[:,:,:,:, ind_iter] = likelihood_grid
        likelihood_grid = like_grid_top

    else:
        like_calc = Parallel(
            n_jobs=50
        )(
            delayed(
                _get_likelihood
            )(
                index,
                xb,
                yb,
                zb,
                rb,
                xs,
                ys,
                zs,
                tau_data,
                n_iter_bub,
                redshift=redshift,
                muv=muv,
                include_muv_unc=include_muv_unc,
                beta_data=beta_data,
                use_EW=use_EW,
                xH_unc=xH_unc,
                la_e=la_e,
                flux_limit=flux_limit,
                like_on_flux=like_on_flux,
            ) for index, (xb, yb, zb, rb) in enumerate(
                itertools.product(x_grid, y_grid, z_grid, r_grid)
            )
        )
        like_calc.sort(key=lambda x: x[0])
        likelihood_grid = np.array([l[1] for l in like_calc])
        likelihood_grid= likelihood_grid.reshape(
            (len(x_grid), len(y_grid), len(z_grid), len(r_grid))
        )

    return likelihood_grid


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mock_direc", type=str, default=None)
    parser.add_argument("--redshift", type=float, default=7.5)
    parser.add_argument("--r_bub", type=float, default=15.0)
    parser.add_argument("--max_dist", type=float, default=15.0)
    parser.add_argument("--n_gal", type=int, default=20)
    parser.add_argument("--obs_pos", type=bool, default=False)
    parser.add_argument("--use_EW", type=bool, default=False)
    parser.add_argument("--diff_mags", type=bool, default=True)
    parser.add_argument(
        "--use_Endsley_Stark_mags",
        type=bool, default=False
    )
    parser.add_argument("--mag_unc", type=bool, default=True)
    parser.add_argument("--xH_unc", type=bool, default=False)
    parser.add_argument("--muv_cut", type=float, default=-19.0)
    parser.add_argument("--diff_pos_prob", type=bool, default=True)
    parser.add_argument("--multiple_iter", type=int, default=None)
    parser.add_argument(
        "--save_dir",
        type=str,
        default='/home/inikolic/projects/Lyalpha_bubbles/code/'
    )
    parser.add_argument("--flux_limit", type=float, default=1e-18)
    parser.add_argument("--uvlf_consistently", type=bool, default=False)
    parser.add_argument("--fluct_level", type=float, default=None)

    parser.add_argument("--like_on_flux", type=bool, default=False)

    inputs = parser.parse_args()

    if inputs.uvlf_consistently:
        if inputs.fluct_level is None:
            raise ValueError("set you density value")
        n_gal = calculate_number(
            fluct_level = inputs.fluct_level,
            x_dim = inputs.max_dist * 2,
            y_dim = inputs.max_dist * 2,
            z_dim = inputs.max_dist * 2,
            redshift = inputs.redshift,
            muv_cut = inputs.muv_cut,
        )
    else:
        n_gal = inputs.n_gal
    print("here is the number of galaxies", n_gal)
    #assert False
    if inputs.diff_mags:
        if inputs.use_Endsley_Stark_mags:
            Muv = get_muv(
                10,
                7.5,
                obs='EndsleyStark21',
            )
        else:
            if inputs.multiple_iter:
                Muv = np.zeros((inputs.multiple_iter, n_gal))
                for index_iter in range(inputs.multiple_iter):
                    Muv[index_iter,:] = get_muv(
                        n_gal = n_gal,
                        redshift = inputs.redshift,
                        muv_cut = inputs.muv_cut
                    )
            else:
                Muv = get_muv(
                    n_gal=n_gal,
                    redshift=inputs.redshift,
                    muv_cut=inputs.muv_cut,
                )
    else:
        if inputs.multiple_iter:
            Muv = -22.0 * np.ones((inputs.multiple_iter, n_gal))
        else:
            Muv = -22.0 * np.ones(n_gal)

    if inputs.use_Endsley_Stark_mags:
        beta = np.array([
            -2.11,
            -2.64,
            -1.95,
            -2.06,
            -1.38,
            -2.77,
            -2.44,
            -2.12,
            -2.59,
            -1.43,
            -2.43,
            -2.02
        ])
    else:
        if inputs.multiple_iter:
            beta = -2.0 * np.ones((inputs.multiple_iter, n_gal))
        else:
            beta = -2.0 * np.ones(n_gal)

    if inputs.mock_direc is None:
        if inputs.multiple_iter:
            td = np.zeros((inputs.multiple_iter, n_gal, 100))
            xd = np.zeros((inputs.multiple_iter, n_gal))
            yd = np.zeros((inputs.multiple_iter, n_gal))
            zd = np.zeros((inputs.multiple_iter, n_gal))
            one_J_arr = np.zeros((inputs.multiple_iter, n_gal, len(wave_em)))
            x_b = []
            y_b = []
            z_b = []
            r_bubs = []
            tau_data_I = np.zeros((inputs.multiple_iter, n_gal))

            for index_iter in range(inputs.multiple_iter):
                tdi, xdi, ydi, zdi, x_bi, y_bi, z_bi, r_bubs_i = get_mock_data(
                    n_gal=n_gal,
                    z_start=inputs.redshift,
                    r_bubble=inputs.r_bub,
                    dist=inputs.max_dist,
                    ENDSTA_data=inputs.obs_pos,
                    diff_pos_prob=inputs.diff_pos_prob,
                )
                td[index_iter, :, :] = tdi
                xd[index_iter, :] = xdi
                yd[index_iter, :] = ydi
                zd[index_iter, :] = zdi
                x_b.append(x_bi)
                y_b.append(y_bi)
                z_b.append(z_bi)
                r_bubs.append(r_bubs_i)
                one_J = get_js(
                    z=inputs.redshift,
                    muv=Muv[index_iter],
                    n_iter=n_gal,
                )
                one_J_arr[index_iter,:,:] = np.array(one_J[0][:n_gal])

     #           print(tau_data_I, np.shape(tau_data_I))
                for i_gal in range(len(tdi)):
                    tau_cgm_gal = tau_CGM(Muv[index_iter][i_gal])
                    eit = np.exp(-tdi[i_gal])
                    tau_data_I[index_iter, i_gal] = np.trapz(
                        eit * tau_cgm_gal * one_J[0][i_gal] / integrate.trapz(
                            one_J[0][i_gal],
                            wave_em.value
                        ), wave_em.value)
      #          print(tau_data_I, np.shape(tau_data_I))

        else:
            td, xd, yd, zd, x_b, y_b, z_b, r_bubs = get_mock_data(
                n_gal=n_gal,
                z_start=inputs.redshift,
                r_bubble=inputs.r_bub,
                dist=inputs.max_dist,
                ENDSTA_data=inputs.obs_pos,
                diff_pos_prob=inputs.diff_pos_prob,
            )
            tau_data_I = []
            one_J = get_js(z=inputs.redshift, muv=Muv, n_iter=len(Muv))
            for i in range(len(td)):
                eit = np.exp(-td[i])
                tau_cgm_gal = tau_CGM(Muv[i])
                tau_data_I.append(
                    np.trapz(
                        eit * tau_cgm_gal * one_J[0][i]/integrate.trapz(
                            one_J[0][i],
                            wave_em.value
                        ),
                        wave_em.value)
                )
        if inputs.use_EW:
            ew_factor, la_e = p_EW(Muv.flatten(), beta.flatten(), return_lum=True)
            #print("This is la_e now", la_e, "this is shape of Muv", np.shape(Muv))
            ew_factor=ew_factor.reshape((np.shape(Muv)))
            la_e=la_e.reshape((np.shape(Muv)))
            #print("and this is it now: ", la_e, "\n with a shape", np.shape(la_e))
            data = np.array(tau_data_I)
        else:
            data = np.array(tau_data_I)

    else:
        data = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/tau_data.npy'  # maybe I'll change this
        )
        xd = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/x_gal_mock.npy'
        )
        yd = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/y_gal_mock.npy'
        )
        zd = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/z_gal_mock.npy'
        )
        x_b = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/x_bub_mock.npy'
        )
        y_b = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/y_bub_mock.npy'
        )
        z_b = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/z_bub_mock.npy'
        )
        r_bubs = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/r_bubs_mock.npy'
        )

        if os.path.isfile(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/tau_shape.npy'
        ):
            td = np.load(
                '/home/inikolic/projects/Lyalpha_bubbles/code/'
                + inputs.mock_direc
                + '/tau_shape.npy'
            )
        if os.path.isfile(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/Muvs.npy'
        ):
            Muv = np.load(
                '/home/inikolic/projects/Lyalpha_bubbles/code/'
                + inputs.mock_direc
                + '/Muvs.npy'
            )
        if os.path.isfile(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/one_J.npy'
        ):
            one_J = np.load(
                '/home/inikolic/projects/Lyalpha_bubbles/code/'
                + inputs.mock_direc
                + '/one_J.npy'
            )
        if os.path.isfile(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/la_e.npy'
        ):
            la_e = np.load(
                '/home/inikolic/projects/Lyalpha_bubbles/code/'
                + inputs.mock_direc
                + '/la_e.npy'
            )
    #print(tau_data_I, np.shape(tau_data_I))
    #print(np.array(data), np.shape(np.array(data)))
    #assert False

    if inputs.like_on_flux:
        #calculate mock flux
        spec_res = wave_Lya.value * (1 + inputs.redshift) / 2700
        bins = np.arange(wave_em.value[0] * (1 + inputs.redshift),
                         wave_em.value[-1] * (1 + inputs.redshift), spec_res)
        wave_em_dig = np.digitize(wave_em.value * (1 + inputs.redshift), bins)
        bins_po = np.append(bins, bins[-1] + spec_res)
        if inputs.multiple_iter:
            flux_noise_mock = np.zeros((inputs.multiple_iter,n_gal, len(bins)))
        else:
            flux_noise_mock = np.zeros((n_gal, len(bins)))
        if not inputs.multiple_iter:
            if not inputs.mock_direc:
                one_J = one_J[0]
            for index_gal in range(n_gal):
                flux_noise_mock[index_gal,:] = [
                    np.trapz(x=wave_em.value[wave_em_dig == i + 1],
                             y=(la_e[index_gal] * one_J[index_gal] * np.exp(
                                 -td[index_gal]
                            ) * tau_CGM(Muv[index_gal]) / (
                                            4 * np.pi * Cosmo.luminosity_distance(
                                    7.5).to(u.cm).value ** 2) / integrate.trapz(
                              one_J[index_gal],
                                wave_em.value)
                                )[wave_em_dig == i + 1]) for i in range(len(bins))
                ]
        else:
            for index_iter in range(inputs.multiple_iter):
                for index_gal in range(n_gal):
                    flux_noise_mock[index_iter,index_gal,:] = [
                        np.trapz(x=wave_em.value[wave_em_dig == i + 1],
                                 y=(la_e[index_iter,index_gal] * one_J_arr[index_iter,index_gal,:] * np.exp(
                                     -td[index_iter,index_gal,:]
                                ) * tau_CGM(Muv[index_iter, index_gal]) / (
                                                4 * np.pi * Cosmo.luminosity_distance(
                                        7.5).to(u.cm).value ** 2) / integrate.trapz(
                                one_J_arr[index_iter,index_gal,:],
                                    wave_em.value)
                                    )[wave_em_dig == i + 1]) for i in range(len(bins))
                    ]
        flux_noise_mock = flux_noise_mock + np.random.normal(
            0,
            1e-19,
            np.shape(flux_noise_mock)
        )

    if inputs.like_on_flux:
        like_on_flux = flux_noise_mock
    else:
        like_on_flux = False

    likelihoods = sample_bubbles_grid(
        tau_data=np.array(data),
        xs=xd,
        ys=yd,
        zs=zd,
        n_iter_bub=30,
        n_grid=5,
        redshift=inputs.redshift,
        muv=Muv,
        include_muv_unc=inputs.mag_unc,
        use_EW=inputs.use_EW,
        beta_data=beta,
        xH_unc=inputs.xH_unc,
        la_e=la_e,
        multiple_iter=inputs.multiple_iter,
        flux_limit=inputs.flux_limit,
        like_on_flux=like_on_flux,
    )
    np.save(
        inputs.save_dir + '/likelihoods.npy',
        likelihoods
    )
    np.save(
        inputs.save_dir + '/x_gal_mock.npy',
        np.array(xd)
    )
    np.save(
        inputs.save_dir + '/y_gal_mock.npy',
        np.array(yd)
    )
    np.save(
        inputs.save_dir + '/z_gal_mock.npy',
        np.array(zd)
    )
    if inputs.multiple_iter:
        max_len_bubs = 0
        for xbi in x_b:
            if len(xbi)>max_len_bubs:
                max_len_bubs = len(xbi)

        x_b_arr = np.zeros((inputs.multiple_iter, max_len_bubs))
        y_b_arr = np.zeros((inputs.multiple_iter, max_len_bubs))
        z_b_arr = np.zeros((inputs.multiple_iter, max_len_bubs))
        r_b_arr = np.zeros((inputs.multiple_iter, max_len_bubs))

        for ind_iter in range(inputs.multiple_iter):
            x_b_arr[ind_iter, :len(x_b[ind_iter])] = x_b[ind_iter]
            y_b_arr[ind_iter, :len(y_b[ind_iter])] = y_b[ind_iter]
            z_b_arr[ind_iter, :len(z_b[ind_iter])] = z_b[ind_iter]
            r_b_arr[ind_iter, :len(r_bubs[ind_iter])] = r_bubs[ind_iter]
    else:
        x_b_arr = np.array(x_b)
        y_b_arr = np.array(y_b)
        z_b_arr = np.array(z_b)
        r_b_arr = np.array(r_bubs)
    print("Where am I saving?")
    print(inputs.save_dir + '/x_bub_mock.npy')
    np.save(
        inputs.save_dir + '/x_bub_mock.npy',
        np.array(x_b_arr)
    )
    np.save(
        inputs.save_dir + '/y_bub_mock.npy',
        np.array(y_b_arr)
    )
    np.save(
        inputs.save_dir + '/z_bub_mock.npy',
        np.array(z_b_arr)
    )
    np.save(
        inputs.save_dir + '/r_bubs_mock.npy',
        np.array(r_b_arr)
    )
    np.save(
        inputs.save_dir + '/data.npy',
        np.array(data),
    )

    flux_to_save = np.zeros(len(Muv.flatten()))
    for i,(xdi,ydi,zdi, tdi, li) in enumerate(zip(
            xd.flatten(), yd.flatten(),zd.flatten(),data.flatten(),la_e.flatten()
    )):
        red_s = z_at_value(
            Cosmo.comoving_distance,
            Cosmo.comoving_distance(inputs.redshift) + zdi * u.Mpc,
            ztol=0.00005
        )

        # calculating fluxes if they are given
        if la_e is not None:
            flux_to_save[i] = li / (
                    4 * np.pi * Cosmo.luminosity_distance(
                red_s).to(u.cm).value ** 2
            )
    np.save(
        inputs.save_dir + '/flux_data.npy',
        np.array(flux_to_save.reshape(np.shape(Muv)))
    )
    if inputs.like_on_flux:
        np.save(
            inputs.save_dir + '/flux_spectrum.npy',
            np.array(like_on_flux)
        )
