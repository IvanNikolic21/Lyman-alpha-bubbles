import sys

import numpy as np
from numpy.linalg import LinAlgError
import argparse
import os
import shutil
from scipy.stats import gaussian_kde

from astropy.cosmology import z_at_value
from astropy import units as u
from astropy.cosmology import Planck18 as Cosmo
import datetime
import itertools
from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity

from venv.galaxy_prop import get_js, get_mock_data, p_EW, L_intr_AH22
from venv.galaxy_prop import get_muv, tau_CGM, calculate_number
from venv.igm_prop import tau_wv
from venv.save import HdF5Saver, HdF5SaverAft, HdF5SaveMocks, HdF5LoadMocks
from venv.helpers import z_at_proper_distance, full_res_flux, perturb_flux, comoving_distance_from_source_Mpc
from venv.speed_up import get_content, calculate_taus_post
from venv.cache import cache_main

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
        beta_data=None,
        la_e_in=None,
        flux_int=None,
        flux_limit=1e-18,
        like_on_flux=False,
        cache_dir='/home/inikolic/projects/Lyalpha_bubbles/_cache/',
        n_inside_tau=50,
        bins_tot=20,
        cache=True,
        like_on_tau_full=False,
        noise_on_the_spectrum=2e-20,
        consistent_noise=True,
        cont_filled=None,
        index_iter=None,
        constrained_prior=False,
        reds_of_galaxies=None,
        dir_name=None,
        main_dir='/home/inikolic/projects/Lyalpha_bubbles/code/Lyman-alpha-bubbles',
        la_e_orig=None,
        prior_on_all=False,
        wave_em_dig_arr=False,
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

    :param beta_data: numpy.array or None.
        UV-slopes for each of the mock galaxies. If None, then a default choice
        of -2.0 is used.
    :param cache_dir: string
        Directory where files will be cached.

    :return:

    Note: to be used only in the sampler
    """

    # Let's define here whatever will be needed for the constrained prior
    # calculation. The only thing to define is the width of the distribution
    # and the rejection criterion
    if constrained_prior:
        width_conp = 0.3

    likelihood_spec = np.zeros((len(xs), bins_tot - 1))
    likelihood_int = np.zeros((len(xs)))
    likelihood_tau = np.zeros((len(xs)))
    if constrained_prior:
        likelihood_spec_cp = np.zeros((len(xs), bins_tot - 1))
        likelihood_int_cp = np.zeros((len(xs)))
    # from now on a likelihood is an array that stores cumulative likelihoods
    # for all galaxies up to a certain number
    taus_tot = []
    flux_tot = []
    spectrum_tot = []

    # For these parameters, let's iterate over galaxies
    if beta_data is None:
        beta_data = np.zeros(len(xs))
    if reds_of_galaxies is None:
        reds_of_galaxies_in = np.zeros(len(xs))
    else:
        reds_of_galaxies_in = reds_of_galaxies

    names_used_this_iter = []

    keep_conp = np.ones((len(xs), n_inside_tau * n_iter_bub), dtype=int)

    for index_gal, (xg, yg, zg, muvi, beti, li) in enumerate(
            zip(xs, ys, zs, muv, beta_data, la_e_in)
    ):
        if index_iter is not None:
            index_gal_eff = index_gal + index_iter * len(xs)
        else:
            index_gal_eff = index_gal
        # defining a dictionary that's going to contain all information about
        # this run for the caching process

        taus_now = []
        xHs_now = []
        j_s_now = []
        lae_now = np.zeros((n_iter_bub * n_inside_tau))
        flux_now = np.zeros((n_iter_bub * n_inside_tau))
        if like_on_flux is not False:
            spectrum_now = np.zeros((n_iter_bub * n_inside_tau, bins_tot - 1,
                                     bins_tot - 1))  # bins_tot is the maximum number of bins
            if consistent_noise:
                spec_res = wave_Lya.value * (1 + redshift) / 2700
                bins = np.arange(wave_em.value[0] * (1 + redshift),
                                 wave_em.value[-1] * (1 + redshift), spec_res)
                max_bins = len(bins)
                flux_to_save = np.zeros((n_iter_bub * n_inside_tau,max_bins))
        tau_now_full = np.zeros((n_iter_bub * n_inside_tau, len(wave_em)))
        if reds_of_galaxies is None:
            red_s = z_at_proper_distance(
                - zg / (1 + redshift) * u.Mpc, redshift
            )
            reds_of_galaxies_in[index_gal] = red_s
        else:
            red_s = reds_of_galaxies_in[index_gal]

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

        #CGM contribution is non-stochastic so it can be outside the loop.
        tau_cgm_in = tau_CGM(muvi, main_dir=main_dir)

        for n in range(n_iter_bub):
            # j_s = get_js(
            #     muv=muvi,
            #     n_iter=n_inside_tau,
            #     include_muv_unc=include_muv_unc,
            #     fwhm_true=fwhm_true,
            # )
            # if xh_unc:
            #     x_H = get_xH(redshift)  # using the central redshift.
            # else:
            #     x_H = 0.65

            # xHs_now.append(x_H)
            # x_outs, y_outs, z_outs, r_bubs = get_bubbles(
            #     x_H,
            #     200
            # )
            # x_bubs_now.append(x_outs)
            # y_bubs_now.append(y_outs)
            # z_bubs_now.append(z_outs)
            # r_bubs_now.append(r_bubs)

            if n == 0 and cache:
                param_name = str(n_iter_bub) + "_" + str(n_inside_tau)
                n_p = f"{xg:.8f}" + "_" + param_name
                fn_original = cache_dir + '/' + dir_name + n_p + '.hdf5'
                pos_n = f"{xb:.2f}" + "_" + f"{yb:.2f}" + '_' + f"{zb:.2f}"
                b_n = n_p + '_' + pos_n + '_' + f"{rb:.2f}" + '.hdf5'
                fn_copy = cache_dir + '/' + dir_name + b_n
                shutil.copyfile(fn_original, fn_copy)

                try:
                    save_cl = HdF5SaverAft(
                        fn_copy
                    )
                except IndexError:
                    print(
                        "Beware, something weird happened with outside bubble",
                    )
                    raise ValueError
            #
            # tau_now_i = calculate_taus_i(
            #     cont_filled.x_bub_out_full[index_gal_eff][n],
            #     cont_filled.y_bub_out_full[index_gal_eff][n],
            #     cont_filled.z_bub_out_full[index_gal_eff][n],
            #     cont_filled.r_bub_out_full[index_gal_eff][n],
            #     red_s,
            #     z_end_bub,
            #     n_iter=n_inside_tau,
            #     dist=dist,
            # )

            tau_now_i = np.copy(cont_filled.tau_prec_full[
                            index_gal_eff
                        ][
                        n * n_inside_tau: (n + 1) * n_inside_tau,:
                        ])
            fi_bu_en_ru_i = np.copy(cont_filled.first_bubble_encounter_redshift_up_full[
                index_gal_eff
            ][
                n * n_inside_tau: (n + 1) * n_inside_tau
            ])
            fi_bu_en_rl_i = np.copy(cont_filled.first_bubble_encounter_redshift_lo_full[
                index_gal_eff
            ][
                n * n_inside_tau: (n + 1) * n_inside_tau
            ])
            fi_bu_en_czu_i = np.copy(cont_filled.first_bubble_encounter_coord_z_up_full[
                index_gal_eff
            ][
                n * n_inside_tau: (n + 1) * n_inside_tau
            ])
            fi_bu_en_czl_i = np.copy(cont_filled.first_bubble_encounter_coord_z_lo_full[
                index_gal_eff
            ][
                n * n_inside_tau: (n + 1) * n_inside_tau
            ])

            tau_now_i += calculate_taus_post(
                red_s,
                z_end_bub,
                fi_bu_en_czu_i,
                fi_bu_en_ru_i,
                fi_bu_en_czl_i,
                fi_bu_en_rl_i,
                n_iter=n_inside_tau,
            )
            del fi_bu_en_czl_i, fi_bu_en_czu_i, fi_bu_en_rl_i, fi_bu_en_ru_i

            tau_sh = np.shape(tau_now_i)

            if np.any(np.any(tau_now_i[:, 30:] - tau_now_i[:, 29:-1] > 0.0,
                             axis=1)):
                inds_inds_rm = np.array(np.where(
                    np.any(tau_now_i[:, 30:] - tau_now_i[:, 29:-1] > 0.0,
                           axis=1))).flatten()
                for indi_rm in inds_inds_rm:

                    shift_sm = np.random.normal(0.0,0.1)
                    dist = comoving_distance_from_source_Mpc(red_s,
                                                             z_end_bub)
                    tau_now_i[indi_rm] = np.clip(tau_wv(
                        wave_em,
                        dist=np.abs(dist),
                        zs=red_s,
                        z_end=5.3,
                        nf=0.65
                    ) + shift_sm, a_min=0.0,a_max=np.inf)


            tau_now_i_fl = tau_now_i.flatten()

            tau_now_i_fl[tau_now_i_fl < 0.0] = np.inf
            tau_now_i = tau_now_i_fl.reshape(tau_sh)
            tau_now_i = np.nan_to_num(tau_now_i, np.inf)
            if np.any(tau_now_i[:, -1] < 0.0):
                dist = comoving_distance_from_source_Mpc(red_s, z_end_bub).value
                tau_now_i[tau_now_i[:, -1] < 0.0] = tau_wv(
                    wave_em,
                    dist=np.abs(dist),
                    zs=red_s,
                    z_end=5.3,
                    nf=0.65
                )
            tau_now_full[n * n_inside_tau:(n + 1) * n_inside_tau, :] = tau_now_i
            eit_l = np.exp(-np.array(tau_now_i))

            res = np.trapz(
                eit_l * tau_cgm_in * cont_filled.j_s_full[index_gal_eff][
                                         n * n_inside_tau: (
                                            n + 1) * n_inside_tau
                                         ],
                wave_em.value
            )

            del tau_now_i


            taus_now.extend(res.tolist())
            area_factor = np.array(
                [
                    np.trapz(
                        cont_filled.j_s_full[index_gal_eff][n * n_inside_tau + i_inside_tau] * tau_cgm_in,
                        wave_em.value
                    ) / np.trapz(
                        cont_filled.j_s_full[index_gal_eff][n * n_inside_tau + i_inside_tau],
                        wave_em.value
                    ) for i_inside_tau in range(n_inside_tau)
                ]
            )
            #print(area_factor)
            try:
                ind_0 = np.where(area_factor < 1e-20)
                area_factor[ind_0] = 1e-5 #doesn't matter, it's going to be multiplied by zero
            except ValueError:
                pass
            lae_now[
                n * n_inside_tau:(n + 1) * n_inside_tau
            ] = cont_filled.la_flux_out_full[index_gal_eff][
                n * n_inside_tau:(n + 1) * n_inside_tau] / area_factor
            flux_now_i = lae_now[
                n * n_inside_tau:(n + 1) * n_inside_tau
            ] * np.array(
                taus_now
            ).flatten(
            )[n * n_inside_tau:(n + 1) * n_inside_tau] * cont_filled.com_fact[
                             index_gal_eff]
            flux_now_i += np.random.normal(0, 5e-20, np.shape(flux_now_i))
            flux_now[n * n_inside_tau:(n + 1) * n_inside_tau] = flux_now_i

            if np.any(np.isnan(flux_now)) or np.any(np.isinf(flux_now)):
                print("Whoa, something bad happened and you have a nan or inf", flush=True)
                print("Ingredients: Lyman-alpha luminosity:",lae_now, flush=True)
                print("tau:", taus_now, flush=True)
                print("area_factor", area_factor, flush=True)
                ind_of_prob = np.where(area_factor==0.0)
                print("j for nan:", cont_filled.j_s_full[index_gal_eff][n * n_inside_tau: (
                                            n + 1) * n_inside_tau
                                         ][ind_of_prob])
                print("tau cgm", tau_cgm_in)
                print("area_factor nans", ind_of_prob, flush=True)
                try:
                    print("indices:", np.where(np.isnan(flux_now)), np.where(np.isinf(flux_now)))
                except ValueError:
                    print("indices:", np.where(np.isinf(flux_now)))
                print("end result", flux_now[np.isnan(flux_now)], flush=True)
                print("end result", flux_now[np.isinf(flux_now)], flush=True)
                print("end result", flux_now, flush=True)
                raise ValueError

            if constrained_prior:
                if prior_on_all:
                    for index_tau_for, fesci in enumerate(
                            res
                    ):
                        if abs(tau_data[index_gal] - fesci) < 0.2:
                            keep_conp[
                                index_gal, n * n_inside_tau + index_tau_for] = 1
                        else:
                            keep_conp[
                                index_gal, n * n_inside_tau + index_tau_for] = 0

                else:
                    if flux_int[index_gal] > 2 * flux_limit:
                        for index_tau_for, lae_i_for in enumerate(
                                cont_filled.la_flux_out_full[
                                    index_gal_eff
                                ][
                                n * n_inside_tau:(n + 1) * n_inside_tau
                                ]
                        ):
                            li_pert = 10 ** (
                                np.log10(
                                    la_e_orig[index_gal]
                                ) + np.random.normal(0.0, 0.2)
                            )

                            if abs(np.log10(lae_i_for) - np.log10(li_pert)) < width_conp:
                                keep_conp[
                                    index_gal, n * n_inside_tau + index_tau_for] = 1
                            else:
                                keep_conp[
                                    index_gal, n * n_inside_tau + index_tau_for] = 0

            #del res
            #del flux_now_i

            j_s_now.extend(cont_filled.j_s_full[index_gal_eff][
                           n * n_inside_tau: (n + 1) * n_inside_tau
                           ])

            if not consistent_noise:
                for bin_i, wav_dig_i in zip(range(2, bins_tot),
                                            wave_em_dig_arr):
                    spectrum_now_i = np.array(
                        [np.trapz(x=wave_em.value[wav_dig_i == i_bin + 1],
                                  y=(lae_now[
                                        n * n_inside_tau:(n + 1) * n_inside_tau
                                    ][:, np.newaxis] * cont_filled.j_s_full[
                                                            index_gal_eff][
                                                        n * n_inside_tau: (
                                                            n + 1
                                                        ) * n_inside_tau
                                                        ] * eit_l * tau_cgm_in[
                                                                    np.newaxis,
                                                                    :
                                                                ] *
                                     cont_filled.com_fact[
                                         index_gal_eff]
                                     )[:, wav_dig_i == i_bin + 1], axis=1) for
                         i_bin
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
                    spectrum_now[n * n_inside_tau:(n + 1) * n_inside_tau,
                    bin_i - 1,
                    :bin_i] = spectrum_now_i.T
                    #del spectrum_now_i
            else:
                continuum_i = (
                        (cont_filled.la_flux_out_full[index_gal_eff][
                        n * n_inside_tau:(n + 1) * n_inside_tau
                        ]/area_factor)[:, np.newaxis] * cont_filled.j_s_full[index_gal_eff][
                                           n * n_inside_tau: (
                                                n + 1) * n_inside_tau
                                           ] * eit_l * tau_cgm_in[
                                                       np.newaxis, :
                                                       ] * cont_filled.com_fact[
                            index_gal_eff]
                )
                full_flux_res_i = full_res_flux(continuum_i, redshift)
                flux_to_save[
                    n * n_inside_tau:(n + 1) * n_inside_tau,
                    :
                ] = full_flux_res_i
                full_flux_res_i += np.random.normal(
                    0,
                    noise_on_the_spectrum,
                    np.shape(full_flux_res_i)
                )
                # flux_to_save[
                #     n * n_inside_tau:(n + 1) * n_inside_tau,
                #     :
                # ] = full_flux_res_i
                #del continuum_i
                for bin_i, wav_dig_i in zip(
                        range(2, bins_tot), wave_em_dig_arr
                ):
                    spectrum_now[n * n_inside_tau:(n + 1) * n_inside_tau,
                    bin_i - 1,
                    :bin_i] = perturb_flux(
                        full_flux_res_i, bin_i
                    )
                #del full_flux_res_i

        if cache:
            if consistent_noise:
                dict_dat_aft = {
                    'tau_full': np.array(taus_now),
                    'flux_integ': flux_now,
                    'mock_spectra': flux_to_save,
                    'la_e_fwmodels': np.array(cont_filled.la_flux_out_full[index_gal_eff]),
                }
            else:
                dict_dat_aft = {
                    'tau_full': np.array(taus_now),
                    'flux_integ': flux_now,
                    'mock_spectra': spectrum_now,
                    'la_e_fwmodels': np.array(cont_filled.la_flux_out_full[index_gal_eff]),
                }

            save_cl.save_data_after(
                xb,
                yb,
                zb,
                rb,
                dict_dat_aft
            )

            names_used_this_iter.append(save_cl.f_name)
            save_cl.close()
            del save_cl, dict_dat_aft
        flux_tot.append(np.array(flux_now).flatten())
        taus_tot.append(np.array(taus_now).flatten())
        spectrum_tot.append(spectrum_now)
    # print("Calculated all the spectra", spectrum_tot, "Shape of spectra", np.shape(spectrum_tot))
    # assert False
    # print(np.shape(taus_tot_b), np.shape(tau_data), flush=True)

    # try:
    taus_tot_b = []
    flux_tot_b = []
    spectrum_tot_b = []
    if constrained_prior:
        taus_tot_cp = []
        flux_tot_cp = []
        spec_tot_cp = []
    for ind_i_gal, (fi, li, speci) in enumerate(
            zip(flux_tot, taus_tot, spectrum_tot)):
        if np.all(np.array(li) < 10000.0):  # maybe unnecessary
            if constrained_prior:
                taus_tot_cp.append(np.array(li)[(keep_conp[ind_i_gal]).astype(bool)])
                flux_tot_cp.append(np.array(fi)[(keep_conp[ind_i_gal]).astype(bool)])
                spec_tot_cp.append(np.array(speci)[(keep_conp[ind_i_gal]).astype(bool)])
                #print(keep_conp[ind_i_gal])
            taus_tot_b.append(li)
            flux_tot_b.append(fi)
            spectrum_tot_b.append(speci)

        #        print(np.shape(taus_tot_b), np.shape(tau_data), flush=True)
        # print(flux_tot_cp, "This is flux_tot_cp", flush=True)
        # print(flux_tot_b, "This is flux_tot_b", flush=True)
    for ind_data, (flux_line, tau_line, spec_line) in enumerate(
            zip(np.array(flux_tot_b), np.array(taus_tot_b),
                np.array(spectrum_tot_b))
    ):
        tau_kde = gaussian_kde((np.array(tau_line)), bw_method=0.15)
        fl_l = np.log10(1e19 * (4e-19 + (np.array(flux_line))))
        if constrained_prior:
            fl_l_cp = np.log10(1e19 * (4e-19 + (np.array(flux_tot_cp[ind_data]))))
            if len(fl_l_cp) == 0:
                print("This is the problem", ind_data, flush=True)
                print(la_e_in[ind_data], flux_int[ind_data], reds_of_galaxies_in[ind_data], flush=True)
            #if ind_data==0:
                #print("Just in case, this is fl_l", fl_l, flux_line, "flux_line as well", flush=True)
        if np.any(np.isnan(fl_l.flatten())) or np.any(np.isinf(fl_l.flatten())):
            ind_nan = np.isnan(fl_l.flatten()).tolist().index(1)
            print("Oops maybe zeros?")
            if ind_nan is not None:
                print(flux_line[ind_nan], flush=True)

            print("This happens for galaxy with index:", ind_data, flush=True)
                #print("and actual problem:", fl_l[ind_nan], flush=True)
            try:
                ind_inf = np.isinf(fl_l.flatten()).tolist().index(1)
                flux_line_list = flux_line.tolist()
                flux_line_list.pop(np.concatenate(ind_nan, ind_inf))
                flux_line = np.array(flux_line_list)
                if ind_inf is not None:
                    print(flux_line[ind_inf], flush=True)
                spec_line_list = spec_line.tolist()
                spec_line_list.pop(np.concatenate(ind_nan, ind_inf))
                spec_line = np.array(spec_line_list)

            except ValueError:
                ind_inf = np.array([])
                flux_line_list = flux_line.tolist()
                flux_line_list.pop(ind_nan)
                flux_line = np.array(flux_line_list)

                spec_line_list = spec_line.tolist()
                spec_line_list.pop(ind_nan)
                spec_line = np.array(spec_line_list)
                #raise ValueError

        flux_kde = gaussian_kde(
            np.log10(1e19 * (4e-19 + (np.array(flux_line)))),
            bw_method=0.15
        )

        if constrained_prior:
            try:
                flux_kde_cp = gaussian_kde(
                    fl_l_cp,
                    bw_method=0.15
                )
            except ValueError:
                print("What is the fl_l_cp:", fl_l_cp)
                print("This is the length", len(fl_l_cp))
                raise ValueError
            print(len(spec_kde), flush=True)
            print(len(list(range(6,len(bins)))), flush=True)
            # like_on_flux = np.array(like_on_flux)
            # print(np.shape(like_on_flux), flush=True)
            # print(ind_data, "index_data", flush=True)
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
            for bin_i in range(2, bins_tot-1):
                try:
                    data_to_get = 5 * np.log10(
                        10**18.7 * (additive_factors[bin_i-2] + 2*spec_line[:, bin_i - 1, np.array(bins_likelihood[bin_i-2])]).T
                    )
                except IndexError:
                    print("This is bin_i", bin_i)
                    print("There was an Index error for some reason:",np.array(bins_likelihood[bin_i-2]) )
                    raise IndexError
                    #print(data_to_get, flush=True)
                    #print("just in case, print", data_to_get[0], flush=True)
                    #print("also", data_to_get[-1], flush=True)
                    # print(spec_line[:,bin_i-1, 1:bin_i], np.shape(spec_line[:,bin_i-1, 1:bin_i]))
                try:
                    #spec_kde = gaussian_kde(data_to_get, bw_method=0.25)
                    spec_kde = KernelDensity(
                        kernel='exponential',
                        bandwidth=0.12
                    ).fit(
                        data_to_get.T
                    )

                except (TypeError, ValueError, LinAlgError):
                    print(np.array(bins_likelihood[bin_i-2]))
                    print("this is the type error", data_to_get, flush=True)
                    print("where=?", np.where(np.isinf(data_to_get)), flush=True)
                    print("problematic values", spec_line.T[np.where(np.isinf(data_to_get))], flush=True)
                    print("Additive factor:", additive_factors[bin_i-2], flush=True)
                    print("where=nan?", np.where(np.isnan(data_to_get)), flush=True)
                    print("problematic values nans", spec_line[:, bin_i - 1, np.array(bins_likelihood[bin_i-2])].T[np.isnan(data_to_get)], flush=True)
                    raise TypeError
                len_bin = len(np.array(bins_likelihood[bin_i-2]))
                data_to_eval = 5 * np.log10(
                    (10**18.7 * (
                            additive_factors[bin_i-2] + 2*like_on_flux[ind_data][
                                    bin_i - 1, np.array(bins_likelihood[bin_i-2])])
                    ).reshape(1,len_bin)
                )
                # likelihood_spec[:ind_data, bin_i - 1] += np.log(
                #     spec_kde.evaluate(
                #         data_to_eval
                #     )
                # )
                likelihood_spec[:ind_data+1, bin_i - 1] += spec_kde.score_samples(
                    data_to_eval
                )

            if constrained_prior:
                for bin_i in range(2, bins_tot-1):
                    try:
                        data_to_get = 5 * np.log10(
                            10**18.7 * (additive_factors[bin_i-2] + 2*spec_tot_cp[ind_data][:, bin_i - 1,
                                        np.array(bins_likelihood[bin_i-2])]).T
                        )
                    except IndexError:
                        print("This is bin_i", bin_i)
                        print("There was an Index error for some reason:",
                            np.array(bins_likelihood[bin_i - 2]))
                        print("all of them:", additive_factors)
                        print("additive factor", additive_factors[bin_i-2])
                        print(spec_tot_cp[ind_data][:, bin_i - 1,
                                        np.array(bins_likelihood[bin_i-2])])
                    #spec_kde = gaussian_kde(data_to_get, bw_method=0.25)

                    spec_kde = KernelDensity(
                        kernel='exponential',
                        bandwidth=0.12
                    ).fit(
                        data_to_get.T
                    )
                    print(np.shape(data_to_get), "This is the shape of constrained prior", flush=True)
                    len_bin = len(np.array(bins_likelihood[bin_i - 2]))
                    data_to_eval = 5 * np.log10(
                        (10**18.7 * (
                                additive_factors[bin_i-2] + 2*like_on_flux[ind_data][
                                        bin_i - 1, np.array(bins_likelihood[bin_i-2])])
                        ).reshape(1,len_bin)
                    )
                    # likelihood_spec_cp[:ind_data, bin_i - 1] += np.log(
                    #     spec_kde.evaluate(
                    #         data_to_eval
                    #     )
                    # )
                    likelihood_spec_cp[:ind_data+1, bin_i - 1] += spec_kde.score_samples(
                        data_to_eval
                    )
                        # except LinAlgError:
                        #     print(len(spec_line), len(spec_tot_cp[ind_data]),
                        #           flush=True)
                        #     print("Lengths")
                        #     print("Lin Alg Error for bin", bin_i)
                        #     print(data_to_get)
                        #     raise ValueError
            #print("This is flux_int", flux_int)
        if flux_int[ind_data] < flux_limit:
                #print("This galaxy failed the tau test, it's flux is",
                #      flux_int[ind_data])

            likelihood_int[:ind_data] += np.log(flux_kde.integrate_box(0.05,
                                                                       np.log10(
                                                                            1e19 * (
                                                                                   4e-19 + flux_limit))))
                #print("It's integrate likelihood is",
                #      flux_kde.integrate_box(0, flux_limit))
        else:
            #print("all good", flux_int[ind_data])
            likelihood_int[:ind_data] += np.log(flux_kde.evaluate(
                np.log10(1e19 * (4e-19 + flux_int[ind_data])))
            )
        if constrained_prior:
            if flux_int[ind_data] < flux_limit:

                likelihood_int_cp[:ind_data] += np.log(
                    flux_kde_cp.integrate_box(0.05,
                                           np.log10(
                                               1e19 * (
                                                       4e-19 + flux_limit))))
            else:
                likelihood_int_cp[:ind_data] += np.log(flux_kde_cp.evaluate(
                    np.log10(1e19 * (4e-19 + flux_int[ind_data])))
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
    # except (LinAlgError, ValueError, TypeError):
    #     likelihood_tau[:ind_data] += -np.inf
    #     likelihood_spec[:ind_data] += -np.inf
    #     likelihood_int[:ind_data] += -np.inf
    #
    #     print("OOps there was value error, let's see why:", flush=True)
    #     print(spec_tot_cp[ind_data], flush=True)
    #     print(tau_data, flush=True)
    #     print(taus_tot_b, flush=True)
    #     raise TypeError


    if not cache:
        names_used_this_iter = None

    if hasattr(likelihood_tau[0], '__len__'):
        if constrained_prior:
            return ndex, (
                likelihood_tau,
                likelihood_int,
                likelihood_spec,
                likelihood_int_cp,
                likelihood_spec_cp
            ), names_used_this_iter
        return ndex, (
            likelihood_tau,
            likelihood_int,
            likelihood_spec
        ), names_used_this_iter
        # return ndex, (
        #     np.array([np.product(li) for li in likelihood_tau]),
        #     np.array([np.product(li) for li in likelihood_int]),
        #     np.array([np.product(li) for li in likelihood_spec])
        # ), names_used_this_iter
    else:
        if constrained_prior:
            return ndex, (
                likelihood_tau, likelihood_int,
                likelihood_spec, likelihood_int_cp,
            likelihood_spec_cp), names_used_this_iter
        return ndex, (
            likelihood_tau, likelihood_int, likelihood_spec), names_used_this_iter


def sample_bubbles_grid(
        tau_data,
        xs,
        ys,
        zs,
        n_iter_bub=100,
        n_grid=10,
        redshift=7.5,
        muv=None,
        beta_data=None,
        la_e=None,
        flux_int=None,
        multiple_iter=False,
        flux_limit=1e-18,
        like_on_flux=False,
        n_inside_tau=50,
        bins_tot=20,
        cache=True,
        like_on_tau_full=False,
        noise_on_the_spectrum=2e-20,
        consistent_noise=True,
        cont_filled=None,
        constrained_prior=False,
        redshifts_of_mocks=None,
        main_dir='/home/inikolic/projects/Lyalpha_bubbles/code/Lyman-alpha-bubbles/',
        cache_dir='/home/inikolic/projects/Lyalpha_bubbles/_cache/',
        la_e_orig=None,
        r_min_grid=5.0,
        r_max_grid=15.0,
        dist_grid_max=5.0,
        no_xy=False,
        prior_on_all=False,
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
    :param beta_data: float,
        beta data.
    :param xh_unc: boolean
        whether to use uncertainty in the underlying neutral fraction in the
        likelihood analysis
    :param la_e: ~np.array or None
        If provided, it's a numpy array consisting of Lyman-alpha total
        luminosity for each mock-galaxy.

    :return likelihood_grid: np.array of shape (N_grid, N_grid, N_grid, N_grid);
        likelihoods for the data on a grid defined above.
    """

    if cache:
        dir_name = 'dir_' + str(
            datetime.datetime.now().date()
        ) + '_' + str(n_iter_bub) + '_' + str(n_inside_tau) + '/'
    else:
        dir_name = None

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
    z_min = -dist_grid_max
    z_max = dist_grid_max
    r_min = r_min_grid
    r_max = r_max_grid
    z_grid = np.linspace(z_min, z_max, n_grid)
    # x_grid = np.linspace(x_min, x_max, n_grid)[5:6]
    # y_grid = np.linspace(y_min, y_max, n_grid)[5:6]
    if no_xy:
        x_grid = np.linspace(-0.0, 0.0, 1)
        y_grid = np.linspace(-0.0, 0.0, 1)
    else:
        x_grid = np.linspace(-dist_grid_max, dist_grid_max, n_grid)
        y_grid = np.linspace(-dist_grid_max, dist_grid_max, n_grid)
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
        if constrained_prior:
            like_grid_int_cp_top = np.zeros(
                (
                    len(x_grid), len(y_grid), len(z_grid), len(r_grid),
                    np.shape(xs)[1],
                    multiple_iter)
            )
            like_grid_spec_cp_top = np.zeros(
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
                n_jobs=25
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
                    beta_data=beta_data[ind_iter],
                    la_e_in=la_e[ind_iter],
                    flux_int=flux_int[ind_iter],
                    flux_limit=flux_limit,
                    like_on_flux=like_on_flux_i,
                    n_inside_tau=n_inside_tau,
                    bins_tot=bins_tot,
                    cache=cache,
                    like_on_tau_full=like_on_tau_full,
                    noise_on_the_spectrum=noise_on_the_spectrum,
                    consistent_noise=consistent_noise,
                    cont_filled=cont_filled,
                    index_iter=ind_iter,
                    constrained_prior=constrained_prior,
                    reds_of_galaxies=redshifts_of_mocks[ind_iter],
                    dir_name=dir_name,
                    main_dir=main_dir,
                    cache_dir=cache_dir,
                    la_e_orig=la_e_orig[ind_iter],
                    prior_on_all=prior_on_all,
                ) for index, (xb, yb, zb, rb) in enumerate(
                    itertools.product(x_grid, y_grid, z_grid, r_grid)
                )
            )
            like_calc.sort(key=lambda x: x[0])
            likelihood_grid_tau = np.array([l[1][0] for l in like_calc])
            likelihood_grid_int = np.array([l[1][1] for l in like_calc])
            likelihood_grid_spec = np.array([l[1][2] for l in like_calc])
            if constrained_prior:
                likelihood_grid_int_cp = np.array([l[1][3] for l in like_calc])
                likelihood_grid_spec_cp = np.array([l[1][4] for l in like_calc])

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
            if constrained_prior:
                likelihood_grid_int_cp = likelihood_grid_int_cp.reshape(
                    (len(x_grid), len(y_grid), len(z_grid), len(r_grid),
                     np.shape(xs)[1])
                )
                likelihood_grid_spec_cp = likelihood_grid_spec_cp.reshape(
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
            if constrained_prior:
                like_grid_int_cp_top[:, :, :, :, :, ind_iter] = likelihood_grid_int_cp
                like_grid_spec_cp_top[:, :, :, :, :, :,
                ind_iter] = likelihood_grid_spec_cp
            names_grid_top.append([l[2] for l in like_calc])

        if constrained_prior:
            likelihood_grid = (
                like_grid_tau_top,
                like_grid_int_top,
                like_grid_spec_top,
                like_grid_int_cp_top,
                like_grid_spec_cp_top,
            )
        else:
            likelihood_grid = (
                like_grid_tau_top, like_grid_int_top, like_grid_spec_top)
        names_grid = names_grid_top

    else:
        like_calc = Parallel(
            n_jobs=25
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
                beta_data=beta_data,
                la_e_in=la_e,
                flux_int=flux_int,
                flux_limit=flux_limit,
                like_on_flux=like_on_flux,
                n_inside_tau=n_inside_tau,
                bins_tot=bins_tot,
                cache=cache,
                like_on_tau_full=like_on_tau_full,
                noise_on_the_spectrum=noise_on_the_spectrum,
                consistent_noise=consistent_noise,
                cont_filled=cont_filled,
                constrained_prior=constrained_prior,
                reds_of_galaxies=redshifts_of_mocks,
                dir_name=dir_name,
                main_dir=main_dir,
                cache_dir=cache_dir,
                la_e_orig=la_e_orig,
                prior_on_all=prior_on_all,
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

        if constrained_prior:
            likelihood_grid_int_cp = np.array([l[1][3] for l in like_calc])
            likelihood_grid_int_cp = likelihood_grid_int_cp.reshape(
                (len(x_grid), len(y_grid), len(z_grid), len(r_grid), len(xs))
            )
            likelihood_grid_spec_cp = np.array([l[1][4] for l in like_calc])
            likelihood_grid_spec_cp = likelihood_grid_spec_cp.reshape(
                (
                    len(x_grid),
                    len(y_grid),
                    len(z_grid),
                    len(r_grid),
                    len(xs),
                    bins_tot - 1
                )
            )
        if constrained_prior:
            likelihood_grid = (
                likelihood_grid_tau,
                likelihood_grid_int,
                likelihood_grid_spec,
                likelihood_grid_int_cp,
                likelihood_grid_spec_cp
            )
        else:
            likelihood_grid = (
                likelihood_grid_tau,
                likelihood_grid_int,
                likelihood_grid_spec
            )

        names_grid = [l[2] for l in like_calc]

    return likelihood_grid, names_grid


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mock_file", type=str, default=None)
    parser.add_argument("--redshift", type=float, default=7.5)
    parser.add_argument("--r_bub", type=float, default=15.0)
    parser.add_argument("--max_dist", type=float, default=15.0)
    parser.add_argument("--r_min", type=float, default=5.0)
    parser.add_argument("--r_max", type=float, default=15.0)
    parser.add_argument("--dist_grid_max", type=float, default=5.0)
    parser.add_argument("--n_gal", type=int, default=20)
    parser.add_argument("--obs_pos", action="store_true")
    parser.add_argument("--diff_mags", action="store_false")
    parser.add_argument(
        "--use_Endsley_Stark_mags",
        type=bool, default=False
    )
    parser.add_argument("--mag_unc", action="store_true")
    parser.add_argument("--xH_unc", action="store_true")
    parser.add_argument("--muv_cut", type=float, default=-19.0)
    parser.add_argument("--diff_pos_prob", action="store_false")
    parser.add_argument("--multiple_iter", type=int, default=None)
    parser.add_argument(
        "--save_dir",
        type=str,
        default='/home/inikolic/projects/Lyalpha_bubbles/code/'
    )
    parser.add_argument("--flux_limit", type=float, default=1e-18)
    parser.add_argument("--uvlf_consistently", action="store_true")
    parser.add_argument("--fluct_level", type=float, default=None)
    parser.add_argument("--no_xy", action="store_true")
    parser.add_argument("--like_on_flux", action="store_false")

    parser.add_argument("--resolution_worsening", type=float, default=1)
    parser.add_argument("--n_inside_tau", type=int, default=50)
    parser.add_argument("--n_iter_bub", type=int, default=50)
    parser.add_argument("--bins_tot", type=int, default=20)
    parser.add_argument("--high_prob_emit", action="store_true")
    parser.add_argument("--use_cache", type=str, default= None)
    parser.add_argument("--cache", action="store_false")
    parser.add_argument("--fwhm_true", action="store_false")
    parser.add_argument("--n_grid", type=int, default=5)

    parser.add_argument("--EW_fixed", action="store_true")
    parser.add_argument("--like_on_tau_full", action="store_true")
    parser.add_argument("--noise_on_the_spectrum", type=float, default=2e-20)
    parser.add_argument("--consistent_noise", action="store_true")
    parser.add_argument("--constrained_prior", action="store_true")
    parser.add_argument("--AH22_model", action="store_true")
    parser.add_argument("--main_dir", type=str, default="/home/inikolic/projects/Lyalpha_bubbles/code/Lyman-alpha-bubbles/")
    parser.add_argument("--cache_dir", type=str, default='/home/inikolic/projects/Lyalpha_bubbles/_cache/')
    parser.add_argument("--gauss_distr", action="store_true")
    parser.add_argument("--Tang_distr", action="store_true")

    parser.add_argument("--prior_on_all", action="store_true")
    inputs = parser.parse_args()

    if inputs.use_cache is not None:
        cache_main(
            inputs.save_dir,
            inputs.flux_limit,
            inputs.n_inside_tau,
            inputs.n_iter_bub,
            inputs.use_cache,
            inputs.mock_file,
            inputs.redshift,
            inputs.bins_tot,
            inputs.constrained_prior,
            inputs.n_grid,
            inputs.multiple_iter,
            inputs.consistent_noise,
            inputs.noise_on_the_spectrum,
            inputs.gauss_distr,
            inputs.Tang_distr,
            inputs.prior_on_all,
            r_min = inputs.r_min,
            r_max = inputs.r_max
        )
        sys.exit(0)
    if inputs.prior_on_all:
        pass
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
        print("Okay, for now it's not acceptable!")
        raise ValueError
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

    if inputs.mock_file is None:
        if inputs.multiple_iter:
            td = np.zeros((inputs.multiple_iter, n_gal, 100))
            xd = np.zeros((inputs.multiple_iter, n_gal))
            yd = np.zeros((inputs.multiple_iter, n_gal))
            zd = np.zeros((inputs.multiple_iter, n_gal))
            one_J_arr = np.zeros((inputs.multiple_iter, n_gal, len(wave_em)))
            area_factor = np.zeros((inputs.multiple_iter, n_gal))
            x_b = []
            y_b = []
            z_b = []
            r_bubs = []
            tau_data_I = np.zeros((inputs.multiple_iter, n_gal))
            redshifts_of_mocks = np.zeros((inputs.multiple_iter, n_gal))
            for index_iter in range(inputs.multiple_iter):
                tdi, xdi, ydi, zdi, x_bi, y_bi, z_bi, r_bubs_i = get_mock_data(
                    n_gal=n_gal,
                    z_start=inputs.redshift,
                    r_bubble=inputs.r_bub,
                    dist=inputs.max_dist,
                    ENDSTA_data=inputs.obs_pos,
                    diff_pos_prob=inputs.diff_pos_prob,
                )
                for index_gal_in in range(len(xdi)):
                    red_s = z_at_proper_distance(
                        - zdi[index_gal_in] / (1 + inputs.redshift) * u.Mpc, inputs.redshift
                    )
                    redshifts_of_mocks[index_iter, index_gal_in] = red_s
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
                tau_cgm_gal = tau_CGM(Muv[index_iter], main_dir=inputs.main_dir)
                area_factor[index_iter, :] = np.array(
                    [
                        np.trapz(
                            one_J_arr[index_iter][i_gal] * tau_cgm_gal[i_gal],
                            wave_em.value
                        ) / np.trapz(
                            one_J_arr[index_iter][i_gal],
                            wave_em.value
                        ) for i_gal in range(n_gal)
                    ]
                )
                for i_gal in range(len(tdi)):
                    tau_cgm_gal = tau_CGM(Muv[index_iter][i_gal], main_dir=inputs.main_dir)
                    eit = np.exp(-tdi[i_gal])
                    tau_data_I[index_iter, i_gal] = np.trapz(
                        eit * tau_cgm_gal * one_J[0][i_gal],
                        wave_em.value
                    )

        else:
            td, xd, yd, zd, x_b, y_b, z_b, r_bubs = get_mock_data(
                n_gal=n_gal,
                z_start=inputs.redshift,
                r_bubble=inputs.r_bub,
                dist=inputs.max_dist,
                ENDSTA_data=inputs.obs_pos,
                diff_pos_prob=inputs.diff_pos_prob,
            )
            redshifts_of_mocks = np.zeros((n_gal))
            for i in range(n_gal):
                red_s = z_at_proper_distance(
                    - zd[i] / (1 + inputs.redshift) * u.Mpc, inputs.redshift
                )
                redshifts_of_mocks[i] = red_s

            tau_data_I = []
            one_J = get_js(
                z=inputs.redshift,
                muv=Muv,
                n_iter=len(Muv),
                fwhm_true=inputs.fwhm_true
            )
            area_factor = np.array(
                [
                    np.trapz(
                        one_J[0][i_gal] * tau_CGM(Muv[i_gal], main_dir=inputs.main_dir),
                        wave_em.value
                    ) / np.trapz(
                        one_J[0][i_gal],
                        wave_em.value
                    ) for i_gal in range(n_gal)
                ]
            )
            for i in range(len(td)):
                eit = np.exp(-td[i])
                tau_cgm_gal = tau_CGM(Muv[i], main_dir=inputs.main_dir)
                tau_data_I.append(
                    np.trapz(
                        eit * tau_cgm_gal * one_J[0][i] ,
                        wave_em.value)
                )

        if inputs.AH22_model:
            la_e = L_intr_AH22(Muv.flatten())
            ew_factor = np.ones((np.shape(Muv)))
        else:
            ew_factor, la_e = p_EW(
                Muv.flatten(),
                beta.flatten(),
                high_prob_emit=inputs.high_prob_emit,
                EW_fixed=inputs.EW_fixed,
                gauss_distr=inputs.gauss_distr,
                Tang_distr=inputs.Tang_distr,
            )
        ew_factor = ew_factor.reshape((np.shape(Muv)))
        ew_factor_orig = np.copy(ew_factor)
        ew_factor /= area_factor
        la_e = la_e.reshape((np.shape(Muv)))
        la_e_orig = np.copy(la_e)
        la_e /= area_factor #new improvement
        data = np.array(tau_data_I)
        #print(area_factor, "This is area factor of mocks")

    else:

        cl_load = HdF5LoadMocks(
            inputs.mock_file,
        )

        data = np.array(cl_load.f['integrated_tau'])
        xd = np.array(cl_load.f['x_gal_mock'])
        yd = np.array(cl_load.f['y_gal_mock'])
        zd = np.array(cl_load.f['z_gal_mock'])
        x_b = np.array(cl_load.f['x_bub_mock'])
        y_b = np.array(cl_load.f['y_bub_mock'])
        z_b = np.array(cl_load.f['z_bub_mock'])
        r_bubs = np.array(cl_load.f['r_bub_mock'])
        td = np.array(cl_load.f['full_tau'])
        Muv = np.array(cl_load.f['Muvs'])
        if inputs.multiple_iter:
            one_J_arr = np.array(cl_load.f['Lyman_alpha_J'])
        else:
            one_J = np.array(cl_load.f['Lyman_alpha_J'])
        la_e = np.array(cl_load.f['Lyman_alpha_lums'])
        la_e_orig = np.array(cl_load.f['Lyman_alpha_lums_orig'])
        flux_spectrum_mock = np.array(cl_load.f['flux_spectrum'])
        flux_tau = np.array(cl_load.f['flux_integrated'])
        flux_nonoise_save = np.array(cl_load.f['flux_no_noise'])
        cl_load.close_file()
        redshifts_of_mocks = np.zeros(n_gal)
        for i in range(n_gal):
            red_s = z_at_proper_distance(
                - zd[i] / (1 + inputs.redshift) * u.Mpc, inputs.redshift
            )
            redshifts_of_mocks[i] = red_s

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
            if inputs.mock_file is None or flux_spectrum_mock is None:
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

                    if not inputs.mock_file:
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
                                         ) * tau_CGM(Muv[index_gal], main_dir=inputs.main_dir) / (
                                                    4 * np.pi * Cosmo.luminosity_distance(
                                                7.5).to(
                                                u.cm).value ** 2)
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
                                            Muv[index_iter, index_gal], main_dir=inputs.main_dir) / (
                                                   4 * np.pi * Cosmo.luminosity_distance(
                                               7.5).to(
                                               u.cm).value ** 2)
                                           )[wav_dig_i == i + 1]) for i in
                                    range(bin_i)
                                ]
                flux_nonoise_save = np.copy(flux_noise_mock)
                flux_noise_mock = flux_noise_mock + np.random.normal(
                    0,
                    inputs.noise_on_the_spectrum,
                    np.shape(flux_noise_mock)
                )
            else:
                flux_noise_mock = flux_spectrum_mock

        else:
            if inputs.mock_file is None:
                if inputs.multiple_iter:
                    flux_noise_mock = np.zeros(
                        (
                            inputs.multiple_iter,
                            n_gal,
                            inputs.bins_tot - 1,
                            inputs.bins_tot - 1)
                    )
                    flux_nonoise_save = []
                    for ind_iter in range(inputs.multiple_iter):
                        continuum = (
                                la_e[ind_iter, :, np.newaxis] * one_J_arr[
                                                                ind_iter, :,
                                                                :] * np.exp(
                            -td[ind_iter]) * tau_CGM(
                            Muv[ind_iter], main_dir=inputs.main_dir) / (
                                                                               4 * np.pi * Cosmo.luminosity_distance(
                                                                           redshifts_of_mocks[
                                                                               ind_iter]
                                                                       ).to(
                                                                           u.cm).value ** 2)[
                                                                       :,
                                                                       np.newaxis]
                        )
                        full_flux_res = full_res_flux(continuum,
                                                      inputs.redshift)
                        flux_nonoise_save.append(np.copy(full_flux_res))

                        full_flux_res += np.random.normal(
                            0,
                            inputs.noise_on_the_spectrum,
                            np.shape(full_flux_res)
                        )
                        for bin_i, wav_dig_i in zip(
                                range(2, inputs.bins_tot - 1), wave_em_dig_arr
                        ):
                            flux_noise_mock[ind_iter, :, bin_i - 1,
                            :bin_i] = perturb_flux(
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
                        Muv, main_dir=inputs.main_dir) / (
                                    4 * np.pi * Cosmo.luminosity_distance(
                                7.5
                            ).to(u.cm).value ** 2)
                    )

                    full_flux_res = full_res_flux(continuum, inputs.redshift)
                    flux_nonoise_save = np.copy(full_flux_res)

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
            else:

                flux_noise_mock = flux_spectrum_mock

    # assert False
    if inputs.like_on_flux:
        like_on_flux = flux_noise_mock
    else:
        like_on_flux = False

    # from now flux is calculated here in the integrated version, and it contains noise
    if inputs.mock_file is None:
        flux_mock = la_e / (
                4 * np.pi * Cosmo.luminosity_distance(
            redshifts_of_mocks).to(u.cm).value ** 2
        )
        flux_tau = flux_mock * tau_data_I
        flux_tau += np.random.normal(0, 5e-20, np.shape(flux_tau))

    #Next part of the code calculates bins for likelihoods
    bins_likelihood = []
    additive_factors = []
    for bin_i_choice in range(2,inputs.bins_tot-1):
        try:
            list_of_indices = [
                np.where(
                    flux_noise_mock[0][i][bin_i_choice-1][:bin_i_choice] > 3 * inputs.noise_on_the_spectrum
                )[0] for i in range(n_gal)
            ]
        except IndexError:
            list_of_indices = [
                np.where(
                    flux_noise_mock[i][bin_i_choice - 1][
                    :bin_i_choice] > 3 * inputs.noise_on_the_spectrum
                )[0] for i in range(n_gal)
            ]
        if len(np.where(
                np.array(
                    [
                        list(
                            np.concatenate(list_of_indices).ravel()
                        ).count(i) for i in range(bin_i_choice)
                    ]
                ) > 0.06*n_gal #6 percent of galaxies should have some emission there
            )[0]) == 0:
            print("For some reason, no bins were selected, check this out:", np.array(
                    [
                        list(
                            np.concatenate(list_of_indices).ravel()
                        ).count(i) for i in range(bin_i_choice)
                    ]
                ))
            print("for bin choice,", bin_i_choice, "and indices", list_of_indices)
            raise ValueError
        bins_likelihood.append(
            np.where(
                np.array(
                    [
                        list(
                            np.concatenate(list_of_indices).ravel()
                        ).count(i) for i in range(bin_i_choice)
                    ]
                ) > 0.06*n_gal
            )[0]#because it's a tuple
        )
        try:
            # additive_factors.append(
            #     10 * np.abs(
            #         np.min(flux_noise_mock[0,:,bin_i_choice-1,:bin_i_choice])
            #     )
            # )
            additive_factors.append(1e-18)
        except IndexError:
            # additive_factors.append(
            #     10 * np.abs(
            #         np.min(flux_noise_mock[:,bin_i_choice-1,:bin_i_choice])
            #     )
            # ) #5 is probably not enough for the noise since I'm multiplying it by 2.
            additive_factors.append(1e-18)

    print("additive factors:", additive_factors)
    #Next part sets up mocks that are going to be necessary for the likelihood
    #calculation. This is the new idea on how to speed up the calculation,
    #calculating whatever can be calculated beforehand



    cont_filled = get_content(
        Muv.flatten(),
        redshifts_of_mocks,
        xd,
        yd,
        zd,
        n_iter_bub=inputs.n_iter_bub,
        n_inside_tau=inputs.n_inside_tau,
        include_muv_unc=inputs.mag_unc,
        fwhm_true=inputs.fwhm_true,
        redshift=inputs.redshift,
        xh_unc=inputs.xH_unc,
        high_prob_emit=inputs.high_prob_emit,
        EW_fixed=inputs.EW_fixed,
        cache=inputs.cache,
        AH22_model=inputs.AH22_model,
        main_dir=inputs.main_dir,
        cache_dir=inputs.cache_dir,
        gauss_distr=inputs.gauss_distr,
        Tang_distr=inputs.Tang_distr,
    )

    likelihoods, names_used = sample_bubbles_grid(
        tau_data=np.array(data),
        xs=xd,
        ys=yd,
        zs=zd,
        n_iter_bub=inputs.n_iter_bub,
        n_grid=inputs.n_grid,
        redshift=inputs.redshift,
        muv=Muv,
        beta_data=beta,
        la_e=la_e,
        flux_int=flux_tau,
        multiple_iter=inputs.multiple_iter,
        flux_limit=inputs.flux_limit,
        like_on_flux=like_on_flux,
        n_inside_tau=inputs.n_inside_tau,
        cache=inputs.cache,
        like_on_tau_full=inputs.like_on_tau_full,
        noise_on_the_spectrum=inputs.noise_on_the_spectrum,
        consistent_noise=inputs.consistent_noise,
        cont_filled=cont_filled,
        redshifts_of_mocks=redshifts_of_mocks,
        bins_tot=inputs.bins_tot,
        main_dir=inputs.main_dir,
        cache_dir=inputs.cache_dir,
        constrained_prior=inputs.constrained_prior,
        la_e_orig = la_e_orig,
        r_min_grid = inputs.r_min,
        r_max_grid = inputs.r_max,
        dist_grid_max = inputs.dist_grid_max,
        no_xy=inputs.no_xy,
        prior_on_all=inputs.prior_on_all,
    )

    dict_to_save_data = dict()
    if isinstance(likelihoods, tuple):
        # np.save(
        #     inputs.save_dir + '/likelihoods_tau.npy',
        #     likelihoods[0]
        # )
        # np.save(
        #     inputs.save_dir + '/likelihoods_int.npy',
        #     likelihoods[1]
        # )
        # np.save(
        #     inputs.save_dir + '/likelihoods_spec.npy',
        #     likelihoods[2]
        # )
        dict_to_save_data['likelihoods_tau'] = likelihoods[0]
        dict_to_save_data['likelihoods_int'] = likelihoods[1]
        dict_to_save_data['likelihoods_spec'] = likelihoods[2]
        if inputs.constrained_prior:
            dict_to_save_data['likelihoods_int_cp'] = likelihoods[3]
            dict_to_save_data['likelihoods_spec_cp'] = likelihoods[4]

    else:
        # np.save(
        #     inputs.save_dir + '/likelihoods.npy',
        #     likelihoods
        # )
        dict_to_save_data['likelihoods'] = likelihoods
    # np.save(
    #     inputs.save_dir + '/x_gal_mock.npy',
    #     np.array(xd)
    # )
    # np.save(
    #     inputs.save_dir + '/y_gal_mock.npy',
    #     np.array(yd)
    # )
    # np.save(
    #     inputs.save_dir + '/z_gal_mock.npy',
    #     np.array(zd)
    # )
    dict_to_save_data['x_gal_mock'] = np.array(xd)
    dict_to_save_data['y_gal_mock'] = np.array(yd)
    dict_to_save_data['z_gal_mock'] = np.array(zd)

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

    # np.save(
    #     inputs.save_dir + '/x_bub_mock.npy',
    #     np.array(x_b_arr)
    # )
    # np.save(
    #     inputs.save_dir + '/y_bub_mock.npy',
    #     np.array(y_b_arr)
    # )
    # np.save(
    #     inputs.save_dir + '/z_bub_mock.npy',
    #     np.array(z_b_arr)
    # )
    # np.save(
    #     inputs.save_dir + '/r_bubs_mock.npy',
    #     np.array(r_b_arr)
    # )
    # np.save(
    #     inputs.save_dir + '/data.npy',
    #     np.array(data),
    # )

    dict_to_save_data['x_bub_mock'] = np.array(x_b_arr)
    dict_to_save_data['y_bub_mock'] = np.array(y_b_arr)
    dict_to_save_data['z_bub_mock'] = np.array(z_b_arr)
    dict_to_save_data['r_bub_mock'] = np.array(r_b_arr)
    dict_to_save_data['integrated_tau'] = np.array(data)

    if inputs.multiple_iter:
        # np.save(
        #     inputs.save_dir + '/one_J.npy',
        #     np.array(one_J_arr),
        # )
        dict_to_save_data['Lyman_alpha_J'] = np.array(one_J_arr)
    else:
        # np.save(
        #     inputs.save_dir + '/one_J.npy',
        #     np.array(one_J),
        # )
        dict_to_save_data['Lyman_alpha_J'] = np.array(one_J)

    # np.save(
    #     inputs.save_dir + '/Muvs.npy',
    #     np.array(Muv),
    # )
    # np.save(
    #     inputs.save_dir + '/la_e_in.npy',
    #     np.array(la_e),
    # )
    # np.save(
    #     inputs.save_dir + '/td.npy',
    #     np.array(td),
    # )
    dict_to_save_data['Muvs'] = np.array(Muv)
    dict_to_save_data['Lyman_alpha_lums'] = np.array(la_e)
    dict_to_save_data['Lyman_alpha_lums_orig'] = np.array(la_e_orig)

    dict_to_save_data['full_tau'] = np.array(td)
    dict_to_save_data['flux_no_noise'] = np.array(flux_nonoise_save)
    # flux_to_save = np.zeros(len(Muv.flatten()))
    # for i, (xdi, ydi, zdi, tdi, li) in enumerate(zip(
    #         xd.flatten(), yd.flatten(), zd.flatten(), data.flatten(),
    #         la_e.flatten()
    # )):
    #     red_s = z_at_value(
    #         Cosmo.comoving_distance,
    #         Cosmo.comoving_distance(inputs.redshift) + zdi * u.Mpc,
    #         ztol=0.00005
    #     )
    #
    #     # calculating fluxes if they are given
    #     if la_e is not None:
    #         flux_to_save[i] = li / (
    #                 4 * np.pi * Cosmo.luminosity_distance(
    #             red_s).to(u.cm).value ** 2
    #         )
    # np.save(
    #     inputs.save_dir + '/flux_data.npy',
    #     np.array(flux_to_save.reshape(np.shape(Muv)))
    # )
    dict_to_save_data['flux_integrated'] = np.array(flux_tau)
    if inputs.like_on_flux:
        # np.save(
        #     inputs.save_dir + '/flux_spectrum.npy',
        #     np.array(like_on_flux)
        # )
        dict_to_save_data['flux_spectrum'] = np.array(like_on_flux)

    cl_save = HdF5SaveMocks(
        n_gal,
        inputs.n_iter_bub,
        inputs.n_inside_tau,
        inputs.save_dir
    )
    cl_save.save_datasets(dict_to_save_data)
    cl_save.close_file()
    if inputs.cache:
        with open(inputs.save_dir + '/names_done.txt', 'w') as f:
            for line in names_used:
                if len(line[0]) > 1:
                    for li in line:
                        f.write(f"{li}\n")
                else:
                    f.write(f"{line}\n")
