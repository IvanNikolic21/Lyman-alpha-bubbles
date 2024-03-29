import numpy as np
from numpy.linalg import LinAlgError
import argparse
import os
from scipy import integrate
from scipy.stats import gaussian_kde

from astropy.cosmology import z_at_value
from astropy import units as u
from astropy.cosmology import Planck18 as Cosmo
import time, datetime
import itertools
from joblib import Parallel, delayed

from venv.galaxy_prop import get_js, get_mock_data, p_EW
from venv.galaxy_prop import get_muv, tau_CGM, calculate_number
from venv.igm_prop import get_bubbles
from venv.igm_prop import calculate_taus_i, get_xH
from venv.save import HdF5Saver
from venv.helpers import z_at_proper_distance, full_res_flux, perturb_flux

wave_em = np.linspace(1214, 1225., 100) * u.Angstrom
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
        use_ew=False,
        xh_unc=False,
        la_e_in=None,
        flux_int=None,
        flux_limit=1e-18,
        like_on_flux=False,
        cache_dir='/home/inikolic/projects/Lyalpha_bubbles/_cache/',
        resolution_worsening=1,
        n_inside_tau=50,
        bins_tot=20,
        high_prob_emit=False,
        cache=True,
        fwhm_true=False,
        EW_fixed=False,
        like_on_tau_full=False,
        noise_on_the_spectrum=2e-20,
        consistent_noise=True,
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
    :param use_ew: boolean
        Whether likelihood calculation is done on flux and other flux-related
        quantities, unlike the False option when the likelihood is calculated
        on transmission directly.
    :param cache_dir: string
        Directory where files will be cached.

    :return:

    Note: to be used only in the sampler
    """
    if cache:
        dir_name = 'dir_' + str(
            datetime.datetime.now().date()
        ) + '_' + str(n_iter_bub) + '_' + str(n_inside_tau) + '/'
    if like_on_flux is not False:
        bins_arr = [
            np.linspace(
                wave_em.value[0] * (1 + redshift),
                wave_em.value[-1] * (1 + redshift),
                bin_i + 1
            ) for bin_i in range(2, bins_tot)
        ]
        wave_em_dig_arr = [
            np.digitize(
                wave_em.value * (1 + redshift),
                bin_i
            ) for bin_i in bins_arr
        ]

    likelihood_spec = np.zeros((len(xs), bins_tot - 1))
    likelihood_int = np.zeros((len(xs)))
    likelihood_tau = np.zeros((len(xs)))
    # from now on likelihood is an array that stores cumulative likelihoods for
    # all galaxies up to a certain number
    com_factor = np.zeros(len(xs))
    taus_tot = []
    flux_tot = []
    j_s_tot = []
    spectrum_tot = []
    if la_e_in is not None:
        flux_mock = np.zeros(len(xs))

    # For these parameters, let's iterate over galaxies
    if beta_data is None:
        beta_data = np.zeros(len(xs))
    reds_of_galaxies = np.zeros(len(xs))
    print(len(xs), "this is the number of galaxies senor", flush=True)

    names_used = []
    for index_gal, (xg, yg, zg, muvi, beti, li) in enumerate(
            zip(xs, ys, zs, muv, beta_data, la_e_in)
    ):

        # defining a dictionary that's going to contain all information about
        # this run for the caching process
        if cache:
            dict_gal = {
                'redshift': redshift,
                'x_galaxy_position': xg,
                'y_galaxy_position': yg,
                'z_galaxy_position': zg,
                'Muv': muvi,
                'beta': beti,
                'Lyman_alpha_lum_galaxy': li,
                'x_bubble_position': xb,
                'y_bubble_position': yb,
                'z_bubble_position': zb,
                'R_main_bubble': rb,
                'n_iter_bub': n_iter_bub,
                'n_inside_tau': n_inside_tau,
            }

        taus_now = []
        flux_now = []
        x_bubs_now = []
        y_bubs_now = []
        z_bubs_now = []
        r_bubs_now = []
        xHs_now = []
        j_s_now = []
        lae_now = np.zeros((n_iter_bub * n_inside_tau))
        flux_now = np.zeros((n_iter_bub * n_inside_tau))
        if like_on_flux is not False:
            spectrum_now = np.zeros((n_iter_bub * n_inside_tau, bins_tot - 1,
                                     bins_tot - 1))  # bins_tot is the maximum number of bins

        tau_now_full = np.zeros((n_iter_bub * n_inside_tau, len(wave_em)))

        red_s = z_at_proper_distance(
            - zg / (1 + redshift) * u.Mpc, redshift
        )
        reds_of_galaxies[index_gal] = red_s
        com_factor[index_gal] = 1 / (4 * np.pi * Cosmo.luminosity_distance(
            redshift).to(u.cm).value ** 2)
        # calculating fluxes if they are given
        if la_e_in is not None:
            flux_mock[index_gal] = li / (
                    4 * np.pi * Cosmo.luminosity_distance(
                red_s).to(u.cm).value ** 2
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
            z_end_bub = z_at_proper_distance(dist / (1 + red_s) * u.Mpc, red_s)
        else:
            z_end_bub = red_s
            dist = 0
        for n in range(n_iter_bub):
            j_s = get_js(
                muv=muvi,
                n_iter=n_inside_tau,
                include_muv_unc=include_muv_unc,
                fwhm_true=fwhm_true,
            )
            if xh_unc:
                x_H = get_xH(redshift)  # using the central redshift.
            else:
                x_H = 0.65

            xHs_now.append(x_H)
            x_outs, y_outs, z_outs, r_bubs = get_bubbles(
                x_H,
                200
            )
            x_bubs_now.append(x_outs)
            y_bubs_now.append(y_outs)
            z_bubs_now.append(z_outs)
            r_bubs_now.append(r_bubs)

            if n == 0 and cache:
                try:
                    save_cl = HdF5Saver(
                        x_gal=xg,
                        x_first_bubble=x_outs[0],
                        output_dir=cache_dir + '/' + dir_name,
                    )
                except IndexError:
                    save_cl = HdF5Saver(
                        x_gal=xg,
                        x_first_bubble=x_outs,
                        output_dir=cache_dir,
                    )
                    print(
                        "Beware, something weird happened with outside bubble",
                        x_outs, y_outs, z_outs
                    )
                save_cl.save_attrs(dict_gal)

            tau_now_i = calculate_taus_i(
                x_outs,
                y_outs,
                z_outs,
                r_bubs,
                red_s,
                z_end_bub,
                n_iter=n_inside_tau,
                dist=dist,
            )
            tau_now_i = np.nan_to_num(tau_now_i, np.inf)
            tau_now_full[n * n_inside_tau:(n + 1) * n_inside_tau, :] = tau_now_i
            eit_l = np.exp(-np.array(tau_now_i))
            tau_cgm_gal_in = tau_CGM(muvi)
            res = np.trapz(
                eit_l * tau_cgm_gal_in * j_s[0] / integrate.trapz(
                    j_s[0],
                    wave_em.value, axis=1)[:, np.newaxis],
                wave_em.value
            )

            if np.all(np.array(res) < 10000):
                if use_ew:
                    taus_now.extend(
                        res.tolist()
                    )
                else:
                    taus_now.extend(np.array(res).tolist())  # just to be sure
            else:
                print("smth wrong", res, flush=True)

            lae_now_i = np.array(
                [p_EW(
                    muvi,
                    beti,
                    high_prob_emit=high_prob_emit,
                    EW_fixed=EW_fixed,
                )[1] for blah in range(len(eit_l))]
            )
            lae_now[n * n_inside_tau:(n + 1) * n_inside_tau] = lae_now_i
            flux_now_i = lae_now_i * np.array(
                taus_now
            ).flatten()[n * n_inside_tau:(n + 1) * n_inside_tau] * com_factor[
                             index_gal]
            flux_now_i += np.random.normal(0, 5e-20, np.shape(flux_now_i))
            flux_now[n * n_inside_tau:(n + 1) * n_inside_tau] = flux_now_i

            j_s_now.extend(j_s[0][:n_inside_tau])
            if not consistent_noise:
                for bin_i, wav_dig_i in zip(range(2, bins_tot), wave_em_dig_arr):
                    spectrum_now_i = np.array(
                        [np.trapz(x=wave_em.value[wav_dig_i == i_bin + 1],
                                  y=(lae_now_i[:, np.newaxis] * j_s[
                                      0] * eit_l * tau_CGM(muvi)[np.newaxis, :] *
                                     com_factor[index_gal] / integrate.trapz(
                                              j_s[0],
                                              wave_em.value, axis=1)[:, np.newaxis]
                                    )[:, wav_dig_i == i_bin + 1], axis=1) for i_bin
                        in range(bin_i)
                        ]
                    )
                    spectrum_now_i += np.random.normal(
                        0,
                        noise_on_the_spectrum,
                        np.shape(spectrum_now_i)
                    )
                # let's investigate properties of this calculation
                #        print(spectrum_now, np.mean(spectrum_now, axis=0), np.mean(spectrum_now,axis=1), np.shape(spectrum_now), np.max(spectrum_now), flush =True)
                #        assert False
                    spectrum_now[n * n_inside_tau:(n + 1) * n_inside_tau, bin_i - 1,
                    :bin_i] = spectrum_now_i.T
            else:
                continuum_i = (
                        lae_now_i[:, np.newaxis] * j_s[0] * eit_l * tau_CGM(
                    muvi)[np.newaxis,:] * com_factor[index_gal] / integrate.trapz(
                                              j_s[0],
                                              wave_em.value, axis=1)[:, np.newaxis]
                )
                full_flux_res_i = full_res_flux(continuum_i, redshift)
                full_flux_res_i += np.random.normal(
                    0,
                    noise_on_the_spectrum,
                    np.shape(full_flux_res_i)
                )
                for bin_i, wav_dig_i in zip(
                        range(2, inputs.bins_tot), wave_em_dig_arr
                ):
                    spectrum_now[n * n_inside_tau:(n + 1) * n_inside_tau, bin_i - 1,
                    :bin_i] = perturb_flux(
                        full_flux_res_i, bin_i
                    )


        if cache:
            max_len = np.max([len(a) for a in x_bubs_now])
            x_bubs_arr = np.zeros((len(x_bubs_now), max_len))
            y_bubs_arr = np.zeros((len(x_bubs_now), max_len))
            z_bubs_arr = np.zeros((len(x_bubs_now), max_len))
            r_bubs_arr = np.zeros((len(x_bubs_now), max_len))
            for i_bub, (xar, yar, zar, rar) in enumerate(
                    zip(x_bubs_now, y_bubs_now, z_bubs_now, r_bubs_now)
            ):
                x_bubs_arr[i_bub, :len(xar)] = xar
                y_bubs_arr[i_bub, :len(xar)] = yar
                z_bubs_arr[i_bub, :len(xar)] = zar
                r_bubs_arr[i_bub, :len(xar)] = rar
            dict_dat = {
                'one_Js': np.array(j_s_now),
                'xHs': np.array(xHs_now),
                'x_bubs_arr': x_bubs_arr,
                'y_bubs_arr': y_bubs_arr,
                'z_bubs_arr': z_bubs_arr,
                'r_bubs_arr': r_bubs_arr,
                'tau_full': tau_now_full,
                'flux_integ': flux_now,
                'Lyman_alpha_iter': lae_now,
                'mock_spectra': spectrum_now,
            }
            save_cl.save_datasets(dict_dat)

            names_used.append(save_cl.fname)
            save_cl.close_file()

        flux_tot.append(np.array(flux_now).flatten())
        taus_tot.append(np.array(taus_now).flatten())
        spectrum_tot.append(spectrum_now)
    # print("Calculated all the spectra", spectrum_tot, "Shape of spectra", np.shape(spectrum_tot))
    # assert False
    taus_tot_b = []
    # print(np.shape(taus_tot_b), np.shape(tau_data), flush=True)

    try:
        taus_tot_b = []
        flux_tot_b = []
        spectrum_tot_b = []
        for fi, li, speci in zip(flux_tot, taus_tot, spectrum_tot):
            if np.all(np.array(li) < 10000.0):  # maybe unnecessary
                taus_tot_b.append(li)
                flux_tot_b.append(fi)
                spectrum_tot_b.append(speci)
        #        print(np.shape(taus_tot_b), np.shape(tau_data), flush=True)

        for ind_data, (flux_line, tau_line, spec_line) in enumerate(
                zip(np.array(flux_tot_b), np.array(taus_tot_b),
                    np.array(spectrum_tot_b))
        ):
            tau_kde = gaussian_kde((np.array(tau_line)), bw_method=0.15)

            flux_kde = gaussian_kde(
                np.log10(1e19 * (3e-19 + (np.array(flux_line)))),
                bw_method=0.15
            )

            # if la_e_in is not None:
            #     flux_tau = flux_mock[ind_data] * tau_data[ind_data]
            # print(len(spec_kde), flush=True)
            # print(len(list(range(6,len(bins)))), flush=True)
            # like_on_flux = np.array(like_on_flux)
            #print(np.shape(like_on_flux), flush=True)
            #print(ind_data, "index_data", flush=True)
            if like_on_tau_full:
                if tau_data[ind_data] < 0.01:
                    likelihood_tau[:ind_data] += np.log(
                        tau_kde.integrate_box(0.0, 0.01)
                    )
                else:
                    likelihood_tau[:ind_data] += np.log(
                        tau_kde.evaluate((tau_data[ind_data]))
                    )

            else:
                if flux_int[ind_data] < flux_limit:
                    pass
                    # likelihood_tau[:ind_data] += np.log(tau_kde.integrate_box(0, 1))
                else:
                    likelihood_tau[:ind_data] += np.log(
                        tau_kde.evaluate((tau_data[ind_data]))
                    )
            if like_on_flux is not False:
                for bin_i in range(2, bins_tot):

                    if bin_i <4:
                        data_to_get = np.log10(
                            1e18 * (5e-19 + spec_line[:, bin_i - 1, 1:bin_i]).T
                        )
                    else:
                        data_to_get = np.log10(
                            1e18 * (5e-19 + spec_line[:, bin_i - 1, 1:4]).T
                        )
                    print(spec_line[:,bin_i-1, 1:bin_i], np.shape(spec_line[:,bin_i-1, 1:bin_i]))
                    spec_kde = gaussian_kde(data_to_get, bw_method=0.2)

                    if bin_i<4:
                        data_to_eval = np.log10(
                                (1e18 * (
                                        5e-19 + like_on_flux[ind_data][
                                                bin_i - 1, 1:bin_i])
                                 ).reshape(bin_i-1, 1)
                            )
                    else:
                        data_to_eval = np.log10(
                            (1e18 * (
                                    5e-19 + like_on_flux[ind_data][
                                            bin_i - 1, 1:4])
                             ).reshape(3, 1)
                        )
                    likelihood_spec[:ind_data, bin_i - 1] += np.log(
                        spec_kde.evaluate(
                            data_to_eval
                        )
                    )
            print("This is flux_int", flux_int)
            if flux_int[ind_data] < flux_limit:
                print("This galaxy failed the tau test, it's flux is",
                      flux_int[ind_data])

                likelihood_int[:ind_data] += np.log(flux_kde.integrate_box(0.05,
                                                                           np.log10(
                                                                               1e19 * (
                                                                                       3e-19 + flux_limit))))
                print("It's integrate likelihood is",
                      flux_kde.integrate_box(0, flux_limit))
            else:
                print("all good", flux_int[ind_data])
                likelihood_int[:ind_data] += np.log(flux_kde.evaluate(
                    np.log10(1e19 * (3e-19 + flux_int[ind_data])))
                )
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
        likelihood_tau[:ind_data] += -np.inf
        likelihood_spec[:ind_data] += -np.inf
        likelihood_int[:ind_data] += -np.inf

        print("OOps there was value error, let's see why:", flush=True)
        #print(tau_data, flush=True)
        #print(taus_tot_b, flush=True)
        raise TypeError

    if not cache:
        names_used = None

    if hasattr(likelihood_tau[0], '__len__'):
        ndex, (likelihood_tau, likelihood_int, likelihood_spec), names_used
        # return ndex, (
        #     np.array([np.product(li) for li in likelihood_tau]),
        #     np.array([np.product(li) for li in likelihood_int]),
        #     np.array([np.product(li) for li in likelihood_spec])
        # ), names_used
    else:
        return ndex, (
            likelihood_tau, likelihood_int, likelihood_spec), names_used


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
        use_ew=False,
        xh_unc=False,
        la_e=None,
        flux_int=None,
        multiple_iter=False,
        flux_limit=1e-18,
        like_on_flux=False,
        resolution_worsening=1,
        n_inside_tau=50,
        bins_tot=20,
        high_prob_emit=False,
        cache=True,
        fwhm_true=False,
        EW_fixed=False,
        like_on_tau_full=False,
        noise_on_the_spectrum=2e-20,
        consistent_noise=True,
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
    :param redshift: float,
        redshift of the analysis.
    :param muv: muv,
        UV magnitude data
    :param include_muv_unc: boolean,
        whether to include muv uncertainty.
    :param beta_data: float,
        beta data.
    :param use_ew: boolean
        whether to use EW or transmissions directly.
    :param xh_unc: boolean
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
    # r_max = 37  # bubble not bigger than the actual size of the box
    r_max = 30
    r_grid = np.linspace(r_min, r_max, n_grid)

    x_min = -15.5
    x_max = 15.5
    x_grid = np.linspace(x_min, x_max, n_grid)

    y_min = -15.5
    y_max = 15.5
    y_grid = np.linspace(y_min, y_max, n_grid)

    # z_min = -12.5
    # z_max = 12.5
    z_min = -5.0
    z_max = 5.0
    r_min = 5.0
    r_max = 15.0
    z_grid = np.linspace(z_min, z_max, n_grid)
    # x_grid = np.linspace(x_min, x_max, n_grid)[5:6]
    # y_grid = np.linspace(y_min, y_max, n_grid)[5:6]
    x_grid = np.linspace(-5.0, 5.0, 5)
    y_grid = np.linspace(-5.0, 5.0, 5)
    r_grid = np.linspace(r_min, r_max, n_grid)
    # print("multiple_iter", multiple_iter, flush=True)
    # assert False
    if multiple_iter:
        like_grid_tau_top = np.zeros(
            (
                len(x_grid), len(y_grid), len(z_grid), len(r_grid),
                np.shape(xs)[1],
                multiple_iter)
        )
        like_grid_int_top = np.zeros(
            (
                len(x_grid), len(y_grid), len(z_grid), len(r_grid),
                np.shape(xs)[1],
                multiple_iter)
        )
        like_grid_spec_top = np.zeros(
            (
                len(x_grid),
                len(y_grid),
                len(z_grid),
                len(r_grid),
                np.shape(xs)[1],
                bins_tot - 1,
                multiple_iter
            )
        )
        names_grid_top = []
        for ind_iter in range(multiple_iter):
            if like_on_flux is not False:
                like_on_flux_i = like_on_flux[ind_iter]
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
                    use_ew=use_ew,
                    xh_unc=xh_unc,
                    la_e_in=la_e[ind_iter],
                    flux_int=flux_int[ind_iter],
                    flux_limit=flux_limit,
                    like_on_flux=like_on_flux_i,
                    resolution_worsening=resolution_worsening,
                    n_inside_tau=n_inside_tau,
                    bins_tot=bins_tot,
                    high_prob_emit=high_prob_emit,
                    cache=cache,
                    fwhm_true=fwhm_true,
                    EW_fixed=EW_fixed,
                    like_on_tau_full=like_on_tau_full,
                    noise_on_the_spectrum=noise_on_the_spectrum,
                    consistent_noise=consistent_noise,
                ) for index, (xb, yb, zb, rb) in enumerate(
                    itertools.product(x_grid, y_grid, z_grid, r_grid)
                )
            )
            like_calc.sort(key=lambda x: x[0])
            likelihood_grid_tau = np.array([l[1][0] for l in like_calc])
            likelihood_grid_int = np.array([l[1][1] for l in like_calc])
            likelihood_grid_spec = np.array([l[1][2] for l in like_calc])

            likelihood_grid_tau = likelihood_grid_tau.reshape(
                (len(x_grid), len(y_grid), len(z_grid), len(r_grid),
                 np.shape(xs)[1])
            )
            likelihood_grid_int = likelihood_grid_int.reshape(
                (len(x_grid), len(y_grid), len(z_grid), len(r_grid),
                 np.shape(xs)[1])
            )
            likelihood_grid_spec = likelihood_grid_spec.reshape(
                (
                    len(x_grid),
                    len(y_grid),
                    len(z_grid),
                    len(r_grid),
                    np.shape(xs)[1],
                    bins_tot - 1
                )
            )
            like_grid_tau_top[:, :, :, :, :, ind_iter] = likelihood_grid_tau
            like_grid_int_top[:, :, :, :, :, ind_iter] = likelihood_grid_int
            like_grid_spec_top[:, :, :, :, :, :,
            ind_iter] = likelihood_grid_spec

            names_grid_top.append([l[2] for l in like_calc])

        likelihood_grid = (
            like_grid_tau_top, like_grid_int_top, like_grid_spec_top)
        names_grid = names_grid_top

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
                use_ew=use_ew,
                xh_unc=xh_unc,
                la_e_in=la_e,
                flux_int=flux_int,
                flux_limit=flux_limit,
                like_on_flux=like_on_flux,
                resolution_worsening=resolution_worsening,
                n_inside_tau=n_inside_tau,
                bins_tot=bins_tot,
                high_prob_emit=high_prob_emit,
                cache=cache,
                fwhm_true=fwhm_true,
                EW_fixed=EW_fixed,
                like_on_tau_full=like_on_tau_full,
                noise_on_the_spectrum=noise_on_the_spectrum,
                consistent_noise=consistent_noise,
            ) for index, (xb, yb, zb, rb) in enumerate(
                itertools.product(x_grid, y_grid, z_grid, r_grid)
            )
        )
        like_calc.sort(key=lambda x: x[0])
        likelihood_grid_tau = np.array([l[1][0] for l in like_calc])
        likelihood_grid_tau = likelihood_grid_tau.reshape(
            (len(x_grid), len(y_grid), len(z_grid), len(r_grid), len(xs))
        )
        likelihood_grid_int = np.array([l[1][1] for l in like_calc])
        likelihood_grid_int = likelihood_grid_int.reshape(
            (len(x_grid), len(y_grid), len(z_grid), len(r_grid), len(xs))
        )
        likelihood_grid_spec = np.array([l[1][2] for l in like_calc])
        likelihood_grid_spec = likelihood_grid_spec.reshape(
            (
                len(x_grid),
                len(y_grid),
                len(z_grid),
                len(r_grid),
                len(xs),
                bins_tot - 1
            )
        )
        likelihood_grid = (
            likelihood_grid_tau,
            likelihood_grid_int,
            likelihood_grid_spec
        )

        names_grid = [l[2] for l in like_calc]

    return likelihood_grid, names_grid


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

    parser.add_argument("--resolution_worsening", type=float, default=1)
    parser.add_argument("--n_inside_tau", type=int, default=50)
    parser.add_argument("--n_iter_bub", type=int, default=50)
    parser.add_argument("--bins_tot", type=int, default=20)
    parser.add_argument("--high_prob_emit", type=bool, default=False)
    parser.add_argument("--cache", type=bool, default=True)
    parser.add_argument("--fwhm_true", type=bool, default=False)
    parser.add_argument("--n_grid", type=int, default=5)

    parser.add_argument("--EW_fixed", type=bool, default=False)
    parser.add_argument("--like_on_tau_full", type=bool, default=False)
    parser.add_argument("--noise_on_the_spectrum", type=float, default=2e-20)
    parser.add_argument("--consistent_noise", type=bool, default=False)
    inputs = parser.parse_args()

    if inputs.uvlf_consistently:
        if inputs.fluct_level is None:
            raise ValueError("set you density value")
        n_gal = calculate_number(
            fluct_level=inputs.fluct_level,
            x_dim=inputs.max_dist * 2,
            y_dim=inputs.max_dist * 2,
            z_dim=inputs.max_dist * 2,
            redshift=inputs.redshift,
            muv_cut=inputs.muv_cut,
        )
    else:
        n_gal = inputs.n_gal
    print("here is the number of galaxies", n_gal)
    # assert False
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
                    Muv[index_iter, :] = get_muv(
                        n_gal=n_gal,
                        redshift=inputs.redshift,
                        muv_cut=inputs.muv_cut
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
                    fwhm_true=inputs.fwhm_true,
                )
                one_J_arr[index_iter, :, :] = np.array(one_J[0][:n_gal])

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
            one_J = get_js(
                z=inputs.redshift,
                muv=Muv,
                n_iter=len(Muv),
                fwhm_true=inputs.fwhm_true
            )
            for i in range(len(td)):
                eit = np.exp(-td[i])
                tau_cgm_gal = tau_CGM(Muv[i])
                tau_data_I.append(
                    np.trapz(
                        eit * tau_cgm_gal * one_J[0][i] / integrate.trapz(
                            one_J[0][i],
                            wave_em.value
                        ),
                        wave_em.value)
                )

        if inputs.use_EW:
            ew_factor, la_e = p_EW(
                Muv.flatten(),
                beta.flatten(),
                return_lum=True,
                high_prob_emit=inputs.high_prob_emit,
                EW_fixed=inputs.EW_fixed,
            )
            # print("This is la_e_in now", la_e_in, "this is shape of Muv", np.shape(Muv))
            ew_factor = ew_factor.reshape((np.shape(Muv)))
            la_e = la_e.reshape((np.shape(Muv)))
            # print("and this is it now: ", la_e_in, "\n with a shape", np.shape(la_e_in))
            data = np.array(tau_data_I)
        else:
            data = np.array(tau_data_I)

    else:
        data = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/tau_data.npy'  # maybe I'll change this
        )[:n_gal]
        xd = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/x_gal_mock.npy'
        )[:n_gal]
        yd = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/y_gal_mock.npy'
        )[:n_gal]
        zd = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/z_gal_mock.npy'
        )[:n_gal]
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
            )[:n_gal]
        if os.path.isfile(
                '/home/inikolic/projects/Lyalpha_bubbles/code/'
                + inputs.mock_direc
                + '/Muvs.npy'
        ):
            Muv = np.load(
                '/home/inikolic/projects/Lyalpha_bubbles/code/'
                + inputs.mock_direc
                + '/Muvs.npy'
            )[:n_gal]
        if os.path.isfile(
                '/home/inikolic/projects/Lyalpha_bubbles/code/'
                + inputs.mock_direc
                + '/one_J.npy'
        ):
            one_J = np.load(
                '/home/inikolic/projects/Lyalpha_bubbles/code/'
                + inputs.mock_direc
                + '/one_J.npy'
            )[:n_gal]
        if os.path.isfile(
                '/home/inikolic/projects/Lyalpha_bubbles/code/'
                + inputs.mock_direc
                + '/la_e_in.npy'
        ):
            la_e = np.load(
                '/home/inikolic/projects/Lyalpha_bubbles/code/'
                + inputs.mock_direc
                + '/la_e_in.npy'
            )[:n_gal]
        if os.path.isfile(
                '/home/inikolic/projects/Lyalpha_bubbles/code/'
                + inputs.mock_direc
                + '/flux_spectrum.npy'
        ):
            flux_spectrum_mock = np.load(
                '/home/inikolic/projects/Lyalpha_bubbles/code/'
                + inputs.mock_direc
                + '/flux_spectrum.npy'
            )
        else:
            flux_spectrum_mock = None
    # print(tau_data_I, np.shape(tau_data_I))
    # print(np.array(data), np.shape(np.array(data)))
    # assert False

    if inputs.like_on_flux:
        bins_arr = [
            np.linspace(
                wave_em.value[0] * (1 + inputs.redshift),
                wave_em.value[-1] * (1 + inputs.redshift),
                bin_i + 1
            ) for bin_i in range(2, inputs.bins_tot)
        ]
        wave_em_dig_arr = [
            np.digitize(
                wave_em.value * (1 + inputs.redshift),
                bin_i
            ) for bin_i in bins_arr
        ]

        # TODO make things self-consistent and modular
        if not inputs.consistent_noise:
            if inputs.mock_direc is None or flux_spectrum_mock is None:
                if inputs.multiple_iter:
                    flux_noise_mock = np.zeros(
                        (
                            inputs.multiple_iter,
                            n_gal,
                            inputs.bins_tot - 1,
                            inputs.bins_tot - 1)
                    )
                else:
                    flux_noise_mock = np.zeros(
                        (n_gal, inputs.bins_tot - 1, inputs.bins_tot - 1)
                    )
                if not inputs.multiple_iter:

                    full_res_flux = 1  # to fill this

                    if not inputs.mock_direc:
                        one_J = one_J[0]

                    for index_gal in range(n_gal):
                        for bin_i, wav_dig_i in zip(
                                range(2, inputs.bins_tot - 1), wave_em_dig_arr
                        ):
                            flux_noise_mock[index_gal, bin_i - 1, :bin_i] = [
                                np.trapz(x=wave_em.value[wav_dig_i == i + 1],
                                         y=(la_e[index_gal] * one_J[
                                             index_gal] * np.exp(
                                             -td[index_gal]
                                         ) * tau_CGM(Muv[index_gal]) / (
                                                    4 * np.pi * Cosmo.luminosity_distance(
                                                7.5).to(
                                                u.cm).value ** 2) / integrate.trapz(
                                             one_J[index_gal],
                                             wave_em.value)
                                            )[wav_dig_i == i + 1]) for i in
                                range(bin_i)
                            ]
                else:
                    for index_iter in range(inputs.multiple_iter):
                        for index_gal in range(n_gal):
                            for bin_i, wav_dig_i in zip(
                                    range(2, inputs.bins_tot - 1),
                                    wave_em_dig_arr
                            ):
                                flux_noise_mock[index_iter, index_gal,
                                bin_i - 1, :bin_i] = [
                                    np.trapz(
                                        x=wave_em.value[wav_dig_i == i + 1],
                                        y=(la_e[
                                               index_iter, index_gal] * one_J_arr[
                                                                        index_iter,
                                                                        index_gal,
                                                                        :] * np.exp(
                                            -td[index_iter, index_gal, :]
                                        ) * tau_CGM(
                                            Muv[index_iter, index_gal]) / (
                                                   4 * np.pi * Cosmo.luminosity_distance(
                                               7.5).to(
                                               u.cm).value ** 2) / integrate.trapz(
                                            one_J_arr[index_iter, index_gal, :],
                                            wave_em.value)
                                           )[wav_dig_i == i + 1]) for i in
                                    range(bin_i)
                                ]
                flux_noise_mock = flux_noise_mock + np.random.normal(
                    0,
                    inputs.noise_on_the_spectrum,
                    np.shape(flux_noise_mock)
                )
            else:
                flux_noise_mock = flux_spectrum_mock

        else:
            if inputs.multiple_iter:
                flux_noise_mock = np.zeros(
                    (
                        inputs.multiple_iter,
                        n_gal,
                        inputs.bins_tot - 1,
                        inputs.bins_tot - 1)
                )
                one_J = one_J_arr[0]
                continuum = (
                        la_e[:, :, np.newaxis] * one_J * np.exp(-td) * tau_CGM(
                    Muv) / (
                                4 * np.pi * Cosmo.luminosity_distance(
                            7.5
                        ).to(u.cm).value ** 2) / integrate.trapz(
                    one_J,
                    wave_em.value,
                    axis=-1,
                )[:, :,np.newaxis]
                )
                full_flux_res = full_res_flux(continuum, inputs.redshift)
                full_flux_res += np.random.normal(
                    0,
                    inputs.noise_on_the_spectrum,
                    np.shape(full_flux_res)
                )
                for bin_i, wav_dig_i in zip(
                        range(2, inputs.bins_tot - 1), wave_em_dig_arr
                ):
                    flux_noise_mock[:,:, bin_i - 1, :bin_i] = perturb_flux(
                        full_flux_res, bin_i
                    )
            else:
                flux_noise_mock = np.zeros(
                    (
                        n_gal,
                        inputs.bins_tot - 1,
                        inputs.bins_tot - 1)
                )
                one_J = one_J[0]
                # print(np.shape(la_e[:, np.newaxis] * one_J[:n_gal,:]), np.shape(td), np.shape(tau_CGM(
                #     Muv)))
                continuum = (
                        la_e[:, np.newaxis] * one_J[:n_gal,:] * np.exp(-td) * tau_CGM(
                    Muv) / (
                                4 * np.pi * Cosmo.luminosity_distance(
                            7.5
                        ).to(u.cm).value ** 2) / integrate.trapz(
                    one_J[:n_gal,:],
                    wave_em.value,
                    axis=-1,
                )[:,  np.newaxis]
                )
                full_flux_res = full_res_flux(continuum, inputs.redshift)
                full_flux_res += np.random.normal(
                    0,
                    inputs.noise_on_the_spectrum,
                    np.shape(full_flux_res)
                )
                for bin_i, wav_dig_i in zip(
                        range(2, inputs.bins_tot), wave_em_dig_arr
                ):
                    flux_noise_mock[:,  bin_i - 1, :bin_i] = perturb_flux(
                        full_flux_res, bin_i
                    )



    # print(np.shape(xd), flush=True)
    # assert False
    if inputs.like_on_flux:
        like_on_flux = flux_noise_mock
    else:
        like_on_flux = False

    # from now flux is calculated here in the integrated version, and it contains noise
    flux_mock = la_e / (
            4 * np.pi * Cosmo.luminosity_distance(
        7.5).to(u.cm).value ** 2
    )
    flux_tau = flux_mock * tau_data_I
    flux_tau += np.random.normal(0, 5e-20, np.shape(flux_tau))

    # print("Finishing setting up mocks", like_on_flux)
    # assert False
    likelihoods, names_used = sample_bubbles_grid(
        tau_data=np.array(data),
        xs=xd,
        ys=yd,
        zs=zd,
        n_iter_bub=inputs.n_iter_bub,
        n_grid=inputs.n_grid,
        redshift=inputs.redshift,
        muv=Muv,
        include_muv_unc=inputs.mag_unc,
        use_ew=inputs.use_EW,
        beta_data=beta,
        xh_unc=inputs.xH_unc,
        la_e=la_e,
        flux_int=flux_tau,
        multiple_iter=inputs.multiple_iter,
        flux_limit=inputs.flux_limit,
        like_on_flux=like_on_flux,
        resolution_worsening=inputs.resolution_worsening,
        n_inside_tau=inputs.n_inside_tau,
        high_prob_emit=inputs.high_prob_emit,
        cache=inputs.cache,
        fwhm_true=inputs.fwhm_true,
        EW_fixed=inputs.EW_fixed,
        like_on_tau_full=inputs.like_on_tau_full,
        noise_on_the_spectrum=inputs.noise_on_the_spectrum,
        consistent_noise=inputs.consistent_noise,
    )
    if isinstance(likelihoods, tuple):
        np.save(
            inputs.save_dir + '/likelihoods_tau.npy',
            likelihoods[0]
        )
        np.save(
            inputs.save_dir + '/likelihoods_int.npy',
            likelihoods[1]
        )
        np.save(
            inputs.save_dir + '/likelihoods_spec.npy',
            likelihoods[2]
        )
    else:
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
            if len(xbi) > max_len_bubs:
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
    if inputs.multiple_iter:
        np.save(
            inputs.save_dir + '/one_J.npy',
            np.array(one_J_arr),
        )
    else:
        np.save(
            inputs.save_dir + '/one_J.npy',
            np.array(one_J),
        )
    np.save(
        inputs.save_dir + '/Muvs.npy',
        np.array(Muv),
    )
    np.save(
        inputs.save_dir + '/la_e_in.npy',
        np.array(la_e),
    )
    np.save(
        inputs.save_dir + '/td.npy',
        np.array(td),
    )
    flux_to_save = np.zeros(len(Muv.flatten()))
    for i, (xdi, ydi, zdi, tdi, li) in enumerate(zip(
            xd.flatten(), yd.flatten(), zd.flatten(), data.flatten(),
            la_e.flatten()
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
    if inputs.cache:
        with open(inputs.save_dir + '/names_done.txt', 'w') as f:
            for line in names_used:
                if len(line[0]) > 1:
                    for li in line:
                        f.write(f"{li}\n")
                else:
                    f.write(f"{line}\n")
