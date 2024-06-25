import numpy as np
from venv.galaxy_prop import get_js, p_EW, L_intr_AH22
from venv.igm_prop import get_xH, get_bubbles
from astropy import units as u
from astropy import constants as const
from venv.helpers import z_at_proper_distance, I, comoving_distance_from_source_Mpc
from venv.igm_prop import tau_wv
from venv.save import HdF5Saver
from joblib import Parallel, delayed
import datetime
from astropy.cosmology import Planck18 as Cosmo

wave_em = np.linspace(1214, 1230., 100) * u.Angstrom
wave_Lya = 1215.67 * u.Angstrom
freq_Lya = (const.c / wave_Lya).to(u.Hz)
r_alpha = 6.25 * 1e8 / (4 * np.pi * freq_Lya.value)


class OutsideContainer:
    def __init__(self):
        self.com_fact = None
        self.j_s_full = []
        self.x_h_full = []
        self.x_bub_out_full = []
        self.y_bub_out_full = []
        self.z_bub_out_full = []
        self.r_bub_out_full = []
        self.la_flux_out_full = []
        self.tau_prec_full = []
        self.first_bubble_encounter_redshift_up_full = []
        self.first_bubble_encounter_redshift_lo_full = []
        self.first_bubble_encounter_coord_z_lo_full = []
        self.first_bubble_encounter_coord_z_up_full = []

    def add_j_s(self, j_s_now):
        self.j_s_full.append(j_s_now)

    def add_xhs(self, x_h_now):
        self.x_h_full.append(x_h_now)

    def add_out_bubble(self, x_out_now, y_out_now, z_out_now, r_out_now):
        self.x_bub_out_full.append(x_out_now)
        self.y_bub_out_full.append(y_out_now)
        self.z_bub_out_full.append(z_out_now)
        self.r_bub_out_full.append(r_out_now)

    def add_la_flux(self, l_flux_now):
        self.la_flux_out_full.append(l_flux_now)

    def add_tau_prec(
            self,
            tau_prec_now,
            first_bubble_encounter_redshift_up_now,
            first_bubble_encounter_redshift_lo_now,
            first_bubble_encounter_coord_z_up_now,
            first_bubble_encounter_coord_z_lo_now,
    ):
        self.tau_prec_full.append(tau_prec_now)
        self.first_bubble_encounter_redshift_up_full.append(
            first_bubble_encounter_redshift_up_now
        )
        self.first_bubble_encounter_redshift_lo_full.append(
            first_bubble_encounter_redshift_lo_now
        )
        self.first_bubble_encounter_coord_z_up_full.append(
            first_bubble_encounter_coord_z_up_now
        )
        self.first_bubble_encounter_coord_z_lo_full.append(
            first_bubble_encounter_coord_z_lo_now
        )

    def add_com_fact(self, com_fact_all):
        self.com_fact = com_fact_all


def get_content(
        Muvs,
        redshifts_of_mocks,
        x_gal_position,
        y_gal_position,
        z_gal_position,
        beta=None,
        n_iter_bub=20,
        n_inside_tau=1000,
        include_muv_unc=False,
        fwhm_true=True,
        redshift=7.5,
        xh_unc=False,
        high_prob_emit=False,
        EW_fixed=False,
        cache=True,
        AH22_model=False,
        cache_dir='/home/inikolic/projects/Lyalpha_bubbles/_cache/',
        main_dir='/home/inikolic/projects/Lyalpha_bubbles/code/Lyman-alpha-bubbles',
        gauss_distr=False,
):
    """
        Function fills up the container which has all of the forward model parts
        that can be computed beforehand. This consists of the galaxy properties
        as well as the outside bubble configuration. The only thing main part of
        the code then needs to do is to implement the main bubble configuration
        and calculate the likelihood.
        Note that n_iter_bub and n_inside_tau have fiducial values that show
        convergence in the integrated flux.
        I can also add here quantities that are going to be accessed throughout
        the likelihood code and that can be calculated once. First thing that
        comes to mind is the comoving distance factor.
    """
    if type(n_iter_bub) is tuple:
        n_iter_bub = n_iter_bub[0]
    if type(n_inside_tau) is tuple:
        n_inside_tau = n_inside_tau[0]

    if cache:
        dir_name = 'dir_' + str(
            datetime.datetime.now().date()
        ) + '_' + str(n_iter_bub) + '_' + str(n_inside_tau) + '/'

    cont_now = OutsideContainer()

    com_factor = np.zeros(len(Muvs.flatten()))
    for index_gal in range(len(Muvs.flatten())):
        com_factor[index_gal] = 1 / (4 * np.pi * Cosmo.luminosity_distance(
            redshifts_of_mocks.flatten()[index_gal]).to(u.cm).value ** 2)
    cont_now.add_com_fact(com_factor.reshape(np.shape(Muvs)))

    if beta is None:
        beta = np.array([-2.0] * len(Muvs.flatten())).reshape(np.shape(Muvs))

    def _get_content_par(muv_i, beti, redi):
        j_s_gal_i = np.zeros((n_iter_bub * n_inside_tau, 100))
        x_h_gal_i = np.zeros(n_iter_bub)
        x_out_gal_i = []
        y_out_gal_i = []
        z_out_gal_i = []
        r_out_gal_i = []
        la_flux_gal_i = np.zeros((n_iter_bub * n_inside_tau))
        tau_prec_i = np.zeros((n_iter_bub * n_inside_tau, 100))
        fi_bu_en_ru_i = np.zeros((n_iter_bub * n_inside_tau))
        fi_bu_en_czu_i = np.zeros((n_iter_bub * n_inside_tau))
        fi_bu_en_rl_i = np.zeros((n_iter_bub * n_inside_tau))
        fi_bu_en_czl_i = np.zeros((n_iter_bub * n_inside_tau))

        for bubble_iter in range(n_iter_bub):
            j_s = get_js(
                muv=muv_i,
                n_iter=n_inside_tau,
                include_muv_unc=include_muv_unc,
                fwhm_true=fwhm_true,
            )
            j_s_gal_i[
            bubble_iter * n_inside_tau: (bubble_iter + 1) * n_inside_tau, :
            ] = j_s[0][:n_inside_tau]
            if xh_unc:
                x_H = get_xH(redshift,main_dir = main_dir)  # using the central redshift.
            else:
                x_H = 0.65
            x_h_gal_i[bubble_iter] = x_H
            x_outs, y_outs, z_outs, r_bubs = get_bubbles(
                x_H,
                200
            )

            x_out_gal_i.append(x_outs)
            y_out_gal_i.append(y_outs)
            z_out_gal_i.append(z_outs)
            r_out_gal_i.append(r_bubs)
            if not AH22_model:
                lae_now_i = np.array(
                    [p_EW(
                        muv_i,
                        beti,
                        high_prob_emit=high_prob_emit,
                        EW_fixed=EW_fixed,
                        gauss_distr=gauss_distr,
                    )[1] for blah in range(n_inside_tau)]
                )
            else:
                lae_now_i = np.array(
                    [L_intr_AH22(muv_i) for blah in range(n_inside_tau)]
                )

            la_flux_gal_i[
            bubble_iter * n_inside_tau: (bubble_iter + 1) * n_inside_tau
            ] = lae_now_i
            (tau_now_i,
             fi_bu_en_czu,
             fi_bu_en_ru,
             fi_bu_en_czl,
             fi_bu_en_rl
             ) = calculate_taus_prep(
                x_small=x_outs,
                y_small=y_outs,
                z_small=z_outs,
                r_bubbles=r_bubs,
                z_source=redi,
                n_iter=n_inside_tau,
            )
            tau_prec_i[
            bubble_iter * n_inside_tau: (bubble_iter + 1) * n_inside_tau, :
            ] = tau_now_i
            fi_bu_en_ru_i[
            bubble_iter * n_inside_tau: (bubble_iter + 1) * n_inside_tau
            ] = fi_bu_en_ru
            fi_bu_en_rl_i[
            bubble_iter * n_inside_tau: (bubble_iter + 1) * n_inside_tau
            ] = fi_bu_en_rl
            fi_bu_en_czu_i[
            bubble_iter * n_inside_tau: (bubble_iter + 1) * n_inside_tau
            ] = fi_bu_en_czu
            fi_bu_en_czl_i[
            bubble_iter * n_inside_tau: (bubble_iter + 1) * n_inside_tau
            ] = fi_bu_en_czl

        return (
            j_s_gal_i,
            x_h_gal_i,
            x_out_gal_i,
            y_out_gal_i,
            z_out_gal_i,
            r_out_gal_i,
            la_flux_gal_i,
            tau_prec_i,
            fi_bu_en_ru_i,
            fi_bu_en_rl_i,
            fi_bu_en_czu_i,
            fi_bu_en_czl_i
        )

    outputs = Parallel(
        n_jobs=30
    )(delayed(
        _get_content_par
    )(
        muv_i,
        beti,
        redi
    ) for (muv_i, beti, redi) in zip(Muvs.flatten(), beta.flatten(), redshifts_of_mocks.flatten()))

    for index_gal in range(len(Muvs.flatten())):

        if cache:
            dict_gal = {
                'redshift': redshifts_of_mocks.flatten()[index_gal],
                'x_galaxy_position': x_gal_position.flatten()[index_gal],
                'y_galaxy_position': y_gal_position.flatten()[index_gal],
                'z_galaxy_position': z_gal_position.flatten()[index_gal],
                'Muv': Muvs.flatten()[index_gal],
                'beta': beta.flatten()[index_gal],
                'n_iter_bub': n_iter_bub,
                'n_inside_tau': n_inside_tau,
            }

            try:
                save_cl = HdF5Saver(
                    x_gal=x_gal_position.flatten()[index_gal],
                    n_iter_bub=n_iter_bub,
                    n_inside_tau=n_inside_tau,
                    output_dir=cache_dir + '/' + dir_name,
                )
            except IndexError:
                save_cl = HdF5Saver(
                    x_gal=x_gal_position.flatten()[index_gal],
                    n_iter_bub=n_iter_bub,
                    n_inside_tau=n_inside_tau,
                    output_dir=cache_dir + '/' + dir_name,
                )
                print(
                    "Beware, something weird happened with outside bubble",
                )
            save_cl.save_attrs(dict_gal)
            max_len = np.max(
                [len(a) for a in outputs[index_gal][2]])
            x_bubs_arr = np.zeros(
                (len(outputs[index_gal][2]), max_len))
            y_bubs_arr = np.zeros(
                (len(outputs[index_gal][2]), max_len))
            z_bubs_arr = np.zeros(
                (len(outputs[index_gal][2]), max_len))
            r_bubs_arr = np.zeros(
                (len(outputs[index_gal][2]), max_len))
            for i_bub, (xar, yar, zar, rar) in enumerate(
                    zip(
                        outputs[index_gal][2],
                        outputs[index_gal][3],
                        outputs[index_gal][4],
                        outputs[index_gal][5],
                    )
            ):
                x_bubs_arr[i_bub, :len(xar)] = np.array(xar).flatten()
                y_bubs_arr[i_bub, :len(xar)] = np.array(yar).flatten()
                z_bubs_arr[i_bub, :len(xar)] = np.array(zar).flatten()
                print(rar, np.shape(rar), np.shape(r_bubs_arr), r_bubs_arr)
                r_bubs_arr[i_bub, :len(xar)] = np.array(rar).flatten()
            dict_dat = {
                #'one_Js': np.array(outputs[index_gal][0]),
                #'xHs': np.array(outputs[index_gal][1]),
                'x_bubs_arr': x_bubs_arr,
                'y_bubs_arr': y_bubs_arr,
                'z_bubs_arr': z_bubs_arr,
                'r_bubs_arr': r_bubs_arr,
                #'tau_prec': np.array(outputs[index_gal][7]),
                #'Lyman_alpha_iter': np.array(outputs[index_gal][6]),
            }
            save_cl.save_datasets(dict_dat)
            save_cl.close_file()

        cont_now.add_j_s(outputs[index_gal][0])
        cont_now.add_la_flux(
            outputs[index_gal][6]
        )
        cont_now.add_tau_prec(
            outputs[index_gal][7],
            outputs[index_gal][8],
            outputs[index_gal][9],
            outputs[index_gal][10],
            outputs[index_gal][11]
        )

    return cont_now


def calculate_taus_prep(
        x_small,
        y_small,
        z_small,
        r_bubbles,
        z_source,
        n_iter=500,
        dist=10,
        prior_end=15,
):
    """
        New way of calculating taus, it's still in the concept. The idea is to
        pre-calculate all tau-s beforehand. Since distribution of taus outside
        doesn't have to correspond to actual situation, i.e. line-of-sights are
        selected randomly. The only information that is going to be used is the
        distance from the bubble. So for z-coordinates that are away from the
        maximum distance of the main bubble I have in my prior.
        For now, it's set to 15cMpc. Note: I'm not using it, we'll see whether
        it's important
    """
    x_small = np.array(x_small)
    y_small = np.array(y_small)
    z_small = np.array(z_small)
    r_bubbles = np.array(r_bubbles).flatten()

    x_random = np.random.uniform(-40, 40, size=n_iter)
    y_random = np.random.uniform(-40, 40, size=n_iter)

    taus = np.zeros((n_iter, len(wave_em)))
    first_bubble_encounter_redshift_up = np.zeros((n_iter))
    first_bubble_encounter_coord_z_up = np.zeros((n_iter))
    first_bubble_encounter_redshift_lo = np.zeros((n_iter))
    first_bubble_encounter_coord_z_lo = np.zeros((n_iter))

    z = wave_em.value / 1215.67 * (1 + z_source) - 1
    one_over_onepz = 1 / (1 + z)
    one_over_onesource = 1 / (1 + z_source)

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

        # || up is the shorter version of the code that was there once searching
        # where small bubbles intersect each sighltine

        if len(z_edge_up) == 0:
            # galaxy doesn't intersect any of the bubbles:

            first_bubble_encounter_coord_z_up[i] = np.inf
            first_bubble_encounter_coord_z_lo[i] = np.inf
            # remember to later check for this
            # to be checked
            continue
        indices_up = z_edge_up.argsort()

        z_edge_up_sorted = z_edge_up[np.flip(indices_up)]
        z_edge_lo_sorted = z_edge_lo[np.flip(indices_up)]
        red_edge_up_sorted = red_edge_up[np.flip(indices_up)]
        red_edge_lo_sorted = red_edge_lo[np.flip(indices_up)]

        while True:
            indices_to_del_lo = []
            indices_to_del_up = []
            for i_fi in range(len(z_edge_up_sorted) - 1):
                if len(z_edge_up_sorted) != 1:
                    if z_edge_lo_sorted[i_fi] < z_edge_up_sorted[i_fi + 1]:
                        # got an overlapping bubble
                        indices_to_del_lo.append(i_fi)
                        indices_to_del_up.append(i_fi + 1)
            if len(indices_to_del_lo) == 0:
                break
            z_edge_lo_sorted = np.delete(z_edge_lo_sorted, indices_to_del_lo)
            z_edge_up_sorted = np.delete(z_edge_up_sorted, indices_to_del_up)
            red_edge_up_sorted = np.delete(red_edge_up_sorted,
                                           indices_to_del_up)
            red_edge_lo_sorted = np.delete(red_edge_lo_sorted,
                                           indices_to_del_lo)
        tau_i = np.zeros(len(wave_em))

        red_edge_up_sorted = np.flip(red_edge_up_sorted)
        z_edge_up_sorted = np.flip(z_edge_up_sorted)
        red_edge_lo_sorted = np.flip(red_edge_lo_sorted)
        z_edge_lo_sorted = np.flip(z_edge_lo_sorted)

        # up to this point, redshifts are calculated for ionized bubbles.
        # An idea is to calculate taus for all outside bubbles, and remember the
        # first encounter with an ionized bubble
        first_bubble_encounter_redshift_up[i] = red_edge_up_sorted[0]
        first_bubble_encounter_coord_z_up[i] = z_edge_up_sorted[0]
        first_bubble_encounter_redshift_lo[i] = red_edge_lo_sorted[0]
        first_bubble_encounter_coord_z_lo[i] = z_edge_lo_sorted[0]

        for index, (z_up_i, z_lo_i, red_up_i, red_lo_i) in enumerate(
                zip(z_edge_up_sorted, z_edge_lo_sorted, red_edge_up_sorted,
                    red_edge_lo_sorted)):
            if index == 0:
                pass
                # The thing is, I'm only gonna consider taus from neutral regions
                # after the first bubble

                # I'm quite confident this is wrong in the main code. However
                # it doesn't change anything since I put on purpose bubbles far
                # from 0.

            elif index != len(z_edge_up_sorted) - 1:
                z_bi = red_edge_lo_sorted[index - 1]
                z_ei = red_up_i
                zb_ar = (1 + z_bi) * one_over_onepz
                tau_i += tau_pref * zb_ar ** 1.5 * (
                        I(zb_ar) - I((1 + z_ei) * one_over_onepz)
                )
            if index == len(z_edge_up_sorted) - 1 and len(
                    z_edge_up_sorted) != 1:
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
            taus[i, :] = tau_i
        except IndexError:
            if n_iter == 1:
                taus = tau_i
            else:
                raise IndexError("Something else")
    taus = taus.flatten()

    return (
        taus.reshape((n_iter, len(wave_em))),
        first_bubble_encounter_coord_z_up,
        first_bubble_encounter_redshift_up,
        first_bubble_encounter_coord_z_lo,
        first_bubble_encounter_redshift_lo,
    )


# Note: for now I'll add
def calculate_taus_post(
        z_source,
        z_end_bubble,
        first_bubble_encounter_coord_z_up,
        first_bubble_encounter_redshift_up,
        first_bubble_encounter_coord_z_lo,
        first_bubble_encounter_redshift_lo,
        n_iter=500,
):
    z = wave_em.value / 1215.67 * (1 + z_source) - 1
    one_over_onepz = 1 / (1 + z)

    tau_gp = 7.16 * 1e5 * ((1 + z_source) / 10) ** 1.5
    tau_pref = tau_gp * r_alpha / np.pi
    taus = np.zeros((n_iter, len(wave_em)))
    for index_iter in range(n_iter):
        z_up_i = first_bubble_encounter_coord_z_up[index_iter]
        z_lo_i = first_bubble_encounter_coord_z_lo[index_iter]
        red_up_i = first_bubble_encounter_redshift_up[index_iter]
        red_lo_i = first_bubble_encounter_redshift_lo[index_iter]
        tau_i = np.zeros((len(wave_em)))
        if z_up_i == np.inf:

            dist = comoving_distance_from_source_Mpc(z_source, z_end_bubble)
            taus[index_iter, :] = tau_wv(
                wave_em,
                dist=dist,
                zs=z_source,
                z_end=5.3,
                nf=0.8
            )
            # no intersections for this iter, already calculated

        else:
            if z_up_i < 0 and z_lo_i < 0:
                print("wrong bubble, it's above the galaxy.")
                raise ValueError
            if z_up_i < 0 < z_lo_i:
                # I think I need to work a bit more on the case where the small
                # ionized bubble intersects the main bubble.
                if red_up_i > z_end_bubble and red_lo_i < z_end_bubble:
                    # already taken into account
                    pass
                elif red_lo_i > z_end_bubble:
                    print("Some big problem, small bubble completely inside big")
                    print(red_lo_i, red_up_i, z_lo_i, z_up_i, z_end_bubble)
                    # raise ValueError(
                    #     "Some big problem, small bubble completely inside big?!"
                    # )
                    # This thing happens, it seems like. I'll investigate in post
                    #processing when it happens. For now, I'll just assume this
                    #is not a significant issue and ignore it.
            else:
                z_bi = z_end_bubble
                z_ei = red_up_i
                zb_ar = (1 + z_bi) * one_over_onepz
                tau_i = tau_pref * zb_ar ** 1.5 * (
                        I(zb_ar) - I((1 + z_ei) * one_over_onepz)
                )
        try:
            taus[index_iter, :] = tau_i
        except IndexError:
            if n_iter == 1:
                taus = tau_i
            else:
                raise IndexError("Something else")
    taus = taus.flatten()
    return taus.reshape((n_iter, len(wave_em)))
