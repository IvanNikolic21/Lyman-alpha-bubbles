#This is a new file that is mean to develop a framework that takes in cached files and simply performs likelihood inference.
#The outline of the get_likelihood will be the same, only the properties will not
#be calculated.

#The first thing to note is that it's the galaxy position that sets up the name
# of the cached file
import numpy as np
import h5py
from venv.save import HdF5LoadMocks
from venv.helpers import z_at_proper_distance, full_res_flux, perturb_flux
from venv.galaxy_prop import tau_CGM, p_EW
from astropy import units as u
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde
import itertools
from scipy.linalg import LinAlgError
from astropy import constants as const
from sklearn.neighbors import KernelDensity
from astropy.cosmology import Planck18 as Cosmo

wave_em = np.linspace(1214, 1225., 100) * u.Angstrom
wave_Lya = 1215.67 * u.Angstrom
freq_Lya = (const.c / wave_Lya).to(u.Hz)
r_alpha = 6.25 * 1e8 / (4 * np.pi * freq_Lya.value)

class HdF5CacheRead:
    def __init__(
            self,
            x_gal,
            n_iter_bub,
            n_inside_tau,
            output_dir,
            x_main,
            y_main,
            z_main,
            r_bub_main,
    ):
        self.f = None
        self.x_gal = x_gal
        self.n_iter_bub = n_iter_bub
        self.n_inside_tau = n_inside_tau
        self.output_dir = output_dir
        pos_n = f"{x_main:.2f}" + "_" + f"{y_main:.2f}" + '_' + f"{z_main:.2f}"
        b_n = pos_n + '_' + f"{r_bub_main:.2f}" + '.hdf5'
        self.f_group_name = str(x_main) + '_' + str(y_main) + '_' + str(z_main) + '_' + str(r_bub_main)
        self.f_name = (self.output_dir +
                      f"{self.x_gal:.8f}" + "_" +
                      f"{self.n_iter_bub}" + "_" +
                      f"{self.n_inside_tau}" + "_" +
                      b_n)
        self.open()

    def open(self):
        self.f = h5py.File(self.f_name, 'r')

    def close(self):
        self.f.close()


class HdF5SaveCached:
    def __init__(
            self,
            n_gal,
            n_iter_bub,
            n_inside_tau,
            save_dir,
    ):
        self.n_gal = n_gal
        self.n_iter_bub = n_iter_bub
        self.n_inside_tau = n_inside_tau
        self.save_dir = save_dir
        self.f_name = self.save_dir + "Init_ngal" + str(
            self.n_gal) + "_nib" + str(
            self.n_iter_bub) + "_nit" + str(
            self.n_inside_tau) + "_cached.hdf5"
        self.__create__()

    def __create__(self,):
        self.f = h5py.File(self.f_name, 'a')

    def save_datasets(self, dict_dat):
        for (nam, val) in dict_dat.items():
            self.f.create_dataset(
                nam,
                dtype="float",
                data=val
            )

    def close_file(self):
        self.f.close()


def get_cache_likelihood(
        x_gal,
        n_iter_bub,
        n_inside_tau,
        x_main,
        y_main,
        z_main,
        R_main,
        output_dir,
        consistent_noise=True,
        noise_on_the_spectrum=2e-20,
        bins_tot=20,
        redshift=7.5,
        constrained_prior=False,
):
    """
    This is to be updated when I think of a function.

    :return:
    """

    f_this = HdF5CacheRead(
        x_gal,
        n_iter_bub,
        n_inside_tau,
        output_dir,
        x_main,
        y_main,
        z_main,
        R_main
    )
    tau_now_full = np.array(f_this.f[f_this.f_group_name]['tau_full'])
    flux_now = np.array(f_this.f[f_this.f_group_name]['flux_integ'])
    #print(flux_now, flush=True)
    if constrained_prior:
        lae_now = np.array(f_this.f[f_this.f_group_name]['la_e_fwmodels'])
    if consistent_noise:
        flux_saved_now = np.array(f_this.f[f_this.f_group_name]['mock_spectra'])
        spectrum_now = np.zeros((n_iter_bub*n_inside_tau, bins_tot - 1, bins_tot-1))
        full_flux_res_i = flux_saved_now + np.random.normal(
            0,
            noise_on_the_spectrum,
            np.shape(flux_saved_now)
        )
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
        for bin_i, wav_dig_i in zip(
                range(2, bins_tot), wave_em_dig_arr
        ):
            spectrum_now[:,
                bin_i - 1,
            :bin_i] = perturb_flux(
                full_flux_res_i, bin_i
            )
    else:
        spectrum_now = np.array(f_this.f[f_this.f_group_name]['mock_spectra'])
    f_this.close()
    if constrained_prior:
        return flux_now, spectrum_now, tau_now_full, lae_now
    return flux_now, spectrum_now, tau_now_full

#I'll also re-write the general structure of the code, but in a much simpler way

def _get_likelihood_cache(
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
        constrained_prior=False,
        reds_of_galaxies=None,
        consistent_noise=True,
        noise_on_the_spectrum=2e-20,
        additive_factors=None,
        bins_likelihood=None,
):
    if constrained_prior:
        width_conp = 0.4
    likelihood_spec = np.zeros((len(xs), bins_tot - 1))
    likelihood_int = np.zeros((len(xs)))
    likelihood_tau = np.zeros((len(xs)))
    taus_tot = []
    flux_tot = []
    spectrum_tot = []
    if beta_data is None:
        beta_data = np.zeros(len(xs))

    keep_conp = np.ones((len(xs), n_inside_tau * n_iter_bub), dtype=int)
    for index_gal, (xg, yg, zg, muvi, beti, li) in enumerate(
            zip(xs, ys, zs, muv, beta_data, la_e_in)
    ):

        full_pack = get_cache_likelihood(
            xg,
            n_iter_bub,
            n_inside_tau,
            xb,
            yb,
            zb,
            rb,
            output_dir=cache_dir,
            consistent_noise=consistent_noise,
            noise_on_the_spectrum=noise_on_the_spectrum,
            bins_tot=bins_tot,
            redshift=redshift,
            constrained_prior=constrained_prior,
        )
        if constrained_prior:
            flux_now, spectrum_now, tau_now_full, lae_now = full_pack
            # if flux_int[index_gal] > flux_limit:
            #     for index_tau_for, res_i_for in enumerate(res):
            #         if abs(res_i_for - tau_data[index_gal]) < width_conp:
            #             keep_conp[
            #                 index_gal, n * n_inside_tau + index_tau_for] = 1
            #         else:
            #             keep_conp[
            #                 index_gal, n * n_inside_tau + index_tau_for] = 0

            if flux_int[index_gal] > 2 * flux_limit:
                for index_tau_for, lae_i_for in enumerate(lae_now):
                    li_pert = 10**(np.log10(li) + np.random.normal(0.0, 0.2))
                    #li_pert = li
                    #print(li, li_pert)

                    if abs(np.log10(lae_i_for) - np.log10(li_pert)) < width_conp:
                        keep_conp[
                            index_gal, index_tau_for] = 1
                    else:
                        keep_conp[
                            index_gal, index_tau_for] = 0
                if np.all(keep_conp[index_gal] == 0):
                    print("This is the case", lae_now, li, flush=True)
                    print(
                        "mean, min, and max",
                        np.min(lae_now[lae_now > 0]),
                        np.max(lae_now), np.mean(lae_now), flush=True)
                    raise ValueError
            #TBC when new updates with constrained prior will be made.
        else:
            flux_now, spectrum_now, tau_now_full = full_pack
        flux_tot.append(np.array(flux_now).flatten())
        taus_tot.append(np.array(tau_now_full).flatten())
        spectrum_tot.append(spectrum_now)
#    try:
    taus_tot_b = []
    flux_tot_b = []
    spectrum_tot_b = []
    for ind_i_gal, (fi, li, speci) in enumerate(
            zip(flux_tot, taus_tot, spectrum_tot)):
            #print(ind_i_gal, fi, li, speci)
            #if np.all(np.array(li) < 10000.0):  # maybe unnecessary
        if constrained_prior:
            taus_tot_b.append(np.array(li)[keep_conp[ind_i_gal].astype(np.bool)])
            flux_tot_b.append(np.array(fi)[keep_conp[ind_i_gal].astype(np.bool)])
            spectrum_tot_b.append(np.array(speci)[keep_conp[ind_i_gal].astype(np.bool)])
        else:
            taus_tot_b.append(li)
            flux_tot_b.append(fi)
            spectrum_tot_b.append(speci)

        #print("Inside likelihoods", np.shape(taus_tot_b), np.shape(tau_data), flush=True)
    if xb==0.0 and yb==0.0 and zb==0.0 and rb==10.0:
        print(xb,yb,zb,rb, flux_tot_b[0][0],flush=True)
    for ind_data, (flux_line, tau_line, spec_line) in enumerate(
            zip(np.array(flux_tot_b), np.array(taus_tot_b),
                np.array(spectrum_tot_b))
    ):
        #tau_kde = gaussian_kde((np.array(tau_line)), bw_method=0.15)
        fl_l = np.log10(1e19 * (4e-19 + (np.array(flux_line))))
            #if ind_data==0:
                #print("Just in case, this is fl_l", fl_l, flux_line, "flux_line as well", flush=True)
        if np.any(np.isnan(fl_l.flatten())) or np.any(np.isinf(fl_l.flatten())):
            print(fl_l.flatten())
            print(np.isnan(fl_l.flatten()).tolist())
            # ind_nan = np.isnan(fl_l.flatten()).tolist().index(1)
            # ind_inf = np.isinf(fl_l.flatten()).tolist().index(1)

            ind_nan = np.isnan(fl_l.flatten()).tolist().index(1)

            print("and actual problem spec:", spec_line[ind_nan])
            print("Tau: ", tau_line[ind_nan])
            print("and actual problem:", np.array(flux_line)[ind_nan], flush=True)

            try:
                ind_inf = np.isinf(fl_l.flatten()).tolist().index(1)
                flux_line_list = flux_line.tolist()
                flux_line_list.pop(np.concatenate(ind_nan, ind_inf))
                flux_line = np.array(flux_line_list)

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

            # spec_line.pop(np.concatenate(ind_nan, ind_inf))
                #raise ValueError

        try:
            flux_kde = gaussian_kde(
                np.log10(1e19 * (4e-19 + (np.array(flux_line)))),
                bw_method=0.15
            )
        except ValueError:
            if constrained_prior:
                print("What is the fl_l_cp:", flux_line, flush=True)
                print("This is the length", len(flux_line), flush=True)
                print("This is the luminosity", la_e_in[ind_data], flush=True)
                raise ValueError
            else:
                print("Don't know why")
                raise ValueError


            # if like_on_tau_full:
            #     if tau_data[ind_data] < 0.01:
            #         likelihood_tau[:ind_data] += np.log(
            #             tau_kde.integrate_box(0.0, 0.01)
            #         )
            #     else:
            #         likelihood_tau[:ind_data] += np.log(
            #             tau_kde.evaluate((tau_data[ind_data]))
            #         )
            #
            # else:
            #     if flux_int[ind_data] < flux_limit:
            #         pass
            #         # likelihood_tau[:ind_data] += np.log(tau_kde.integrate_box(0, 1))
            #     else:
            #         likelihood_tau[:ind_data] += np.log(
            #             tau_kde.evaluate((tau_data[ind_data]))
            #         )
        #print(spec_line, flush=True)
        if like_on_flux is not False:
            for bin_i in range(2, bins_tot-1):
                # if bin_i < 7:
                #     data_to_get = 5*np.log10(
                #         10**18.7 * (9e-19 + 2*spec_line[:, bin_i - 1, 1:bin_i]).T
                #     )
                # else:
                data_to_get = 10 * np.log10(
                    10**18.7 * (additive_factors[bin_i-2] *1.6 + 3*spec_line[:, bin_i - 1, np.array(bins_likelihood[bin_i-2])]).T
                )
                if np.any(np.isnan(data_to_get.flatten())):
                    print(np.shape(data_to_get), flush=True)
                    print(np.shape(np.isnan(data_to_get)), flush=True)
                    print(np.shape(spec_line[:, bin_i - 1, np.array(bins_likelihood[bin_i-2])].T), flush=True)
                    try:
                        print("For this galaxy a nan:",spec_line[:, bin_i - 1,np.array(bins_likelihood[bin_i-2])].T[np.isnan(data_to_get)], flush=True )
                        print("There was a nan:", data_to_get[np.isnan(data_to_get)], flush=True)
                    except TypeError:
                        raise TypeError
                if np.any(np.isinf(data_to_get.flatten())):
                    print("There was infinity:", data_to_get[np.isinf(data_to_get)])
                #spec_kde = gaussian_kde(data_to_get, bw_method=0.13)
                spec_kde = KernelDensity(
                    #kernel='epanechnikov',
                    kernel='exponential',
                    bandwidth=0.10
                ).fit(
                    data_to_get.T
                )
                len_bin = len(np.array(bins_likelihood[bin_i-2]))

                # if bin_i < 7:
                data_to_eval = 10 * np.log10(
                        (10**18.7 * (
                            additive_factors[bin_i-2]*1.6 + 3*like_on_flux[ind_data][
                                    bin_i - 1, np.array(bins_likelihood[bin_i-2])])
                        ).reshape(1,len_bin)
                )
                # else:
                #     data_to_eval = 5*np.log10(
                #         (10**18.7 * (
                #                 9e-19 + 2*like_on_flux[ind_data][
                #                         bin_i - 1, 2:7])
                #         ).reshape(1,5)
                #     )
                # likelihood_spec[:ind_data, bin_i - 1] += np.log(
                #     spec_kde.evaluate(
                #         data_to_eval
                #     )
                # )
                likelihood_spec[:ind_data+1, bin_i - 1] += spec_kde.score_samples(
                    data_to_eval
                )
                if spec_kde.score_samples(
                    data_to_eval
                ) == -np.inf:
                    print("There was an inf", data_to_eval, flush=True)
                    print("This is the data to get", data_to_get, flush=True)

        if flux_int[ind_data] < flux_limit:
            likelihood_int[:ind_data+1] += np.log(flux_kde.integrate_box(0.05,
                                                                        np.log10(
                                                                            1e19 * (
                                                                                    4e-19 + flux_limit))))

        else:
            likelihood_int[:ind_data+1] += np.log(flux_kde.evaluate(
                np.log10(1e19 * (4e-19 + flux_int[ind_data])))
            )

    # except (LinAlgError, ValueError, TypeError):
    #     likelihood_tau[:ind_data] += -np.inf
    #     likelihood_spec[:ind_data] += -np.inf
    #     likelihood_int[:ind_data] += -np.inf
    #
    #     print("OOps there was value error, let's see why:", flush=True)

    if hasattr(likelihood_tau[0], '__len__'):
        return ndex, (
            likelihood_tau,
            likelihood_int,
            likelihood_spec
        )

    else:
        return ndex, (
            likelihood_tau, likelihood_int, likelihood_spec)


def cache_main(
    save_dir,
    flux_limit,
    n_inside_tau,
    n_iter_bub,
    use_cache,
    mock_file,
    redshift,
    bins_tot,
    constrained_prior,
    n_grid,
    mult_iter,
    consistent_noise,
    noise_on_the_spectrum,
    gauss_distr=False,
    r_min=5.0,
    r_max=15.0,
):
    if mock_file is None:
        raise ValueError("You need to specify a mock that created this. For now")

    cl_load = HdF5LoadMocks(
            mock_file,
        )

    data = np.array(cl_load.f['integrated_tau'])
    xd = np.array(cl_load.f['x_gal_mock'])
    yd = np.array(cl_load.f['y_gal_mock'])
    zd = np.array(cl_load.f['z_gal_mock'])
    Muv = np.array(cl_load.f['Muvs'])
    if mult_iter is not None:
        n_gal = np.shape(Muv)[1]
    else:
        n_gal = len(Muv)

    if not gauss_distr:
        la_e = np.array(cl_load.f['Lyman_alpha_lums_orig'])
    else:
        one_J_arr = np.array(cl_load.f['Lyman_alpha_J'])
        td = np.array(cl_load.f['full_tau'])
        int_tau = np.array(cl_load.f['integrated_tau'])
        beta = -2.0 * np.ones(np.shape(Muv))
        ew_factor, la_e_orig = p_EW(
            Muv.flatten(),
            beta.flatten(),
            gauss_distr=gauss_distr
        )
        if mult_iter:
            area_factor = np.zeros((mult_iter, n_gal))

            for index_iter in range(mult_iter):
                tau_cgm_gal = tau_CGM(Muv[index_iter])
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
        else:
            area_factor = np.array(
                [
                    np.trapz(
                        one_J_arr[0][i_gal] * tau_CGM(Muv[i_gal]),
                        wave_em.value
                    ) / np.trapz(
                        one_J_arr[0][i_gal],
                        wave_em.value
                    ) for i_gal in range(n_gal)
                ]
            )
        la_e_comp = la_e_orig / area_factor.flatten()
        la_e_comp = la_e_comp.reshape((np.shape(Muv)))
        la_e_orig = la_e_orig.reshape((np.shape(Muv)))
        la_e = np.copy(la_e_orig)
    redshifts_of_mocks = np.zeros(np.product(np.shape(Muv)))
    for i in range(len(redshifts_of_mocks)):
        red_s = z_at_proper_distance(
            - zd.flatten()[i] / (1 + redshift) * u.Mpc, redshift
        )
        redshifts_of_mocks[i] = red_s
    redshifts_of_mocks = redshifts_of_mocks.reshape(np.shape(Muv))
    if not gauss_distr:
        flux_spectrum_mock = np.array(cl_load.f['flux_spectrum'])
        flux_tau = np.array(cl_load.f['flux_integrated'])
    else:
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
        if mult_iter:
            flux_spectrum_mock = np.zeros(
                (
                    mult_iter,
                    n_gal,
                    bins_tot - 1,
                    bins_tot - 1)
            )
            flux_nonoise_save = []
            for ind_iter in range(mult_iter):
                continuum = (
                        la_e_comp[ind_iter, :, np.newaxis] * one_J_arr[
                                                        ind_iter, :,
                                                        :] * np.exp(
                    -td[ind_iter]) * tau_CGM(
                    Muv[ind_iter]) / (
                                                                       4 * np.pi * Cosmo.luminosity_distance(
                                                                   redshifts_of_mocks[
                                                                       ind_iter]
                                                               ).to(
                                                                   u.cm).value ** 2)[
                                                               :,
                                                               np.newaxis]
                )
                full_flux_res = full_res_flux(continuum,
                                              redshift)
                flux_nonoise_save.append(np.copy(full_flux_res))

                full_flux_res += np.random.normal(
                    0,
                    noise_on_the_spectrum,
                    np.shape(full_flux_res)
                )
                for bin_i, wav_dig_i in zip(
                        range(2, bins_tot - 1), wave_em_dig_arr
                ):
                    flux_spectrum_mock[ind_iter, :, bin_i - 1,
                    :bin_i] = perturb_flux(
                        full_flux_res, bin_i
                    )

        else:
            flux_spectrum_mock = np.zeros(
                (
                    n_gal,
                    bins_tot - 1,
                    bins_tot - 1)
            )
            one_J = one_J_arr[0]
            # print(np.shape(la_e[:, np.newaxis] * one_J[:n_gal,:]), np.shape(td), np.shape(tau_CGM(
            #     Muv)))
            continuum = (
                    la_e_comp[:, np.newaxis] * one_J[:n_gal, :] * np.exp(
                -td) * tau_CGM(
                Muv) / (
                            4 * np.pi * Cosmo.luminosity_distance(
                        7.5
                    ).to(u.cm).value ** 2)
            )

            full_flux_res = full_res_flux(continuum, redshift)
            flux_nonoise_save = np.copy(full_flux_res)

            full_flux_res += np.random.normal(
                0,
                noise_on_the_spectrum,
                np.shape(full_flux_res)
            )
            for bin_i, wav_dig_i in zip(
                    range(2, bins_tot), wave_em_dig_arr
            ):
                flux_spectrum_mock[:, bin_i - 1, :bin_i] = perturb_flux(
                    full_flux_res, bin_i
                )
        flux_mock = la_e_comp / (
                4 * np.pi * Cosmo.luminosity_distance(
            redshifts_of_mocks).to(u.cm).value ** 2
        )
        flux_tau = flux_mock * int_tau
        flux_tau += np.random.normal(0, 5e-20, np.shape(flux_tau))

    #this part of code calculates flux if noise is different
    if not gauss_distr and noise_on_the_spectrum is not 2e-20:
        la_e_comp = np.array(cl_load.f['Lyman_alpha_lums'])
        one_J_arr = np.array(cl_load.f['Lyman_alpha_J'])
        td = np.array(cl_load.f['full_tau'])
        int_tau = np.array(cl_load.f['integrated_tau'])
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
        if mult_iter:
            flux_spectrum_mock = np.zeros(
                (
                    mult_iter,
                    n_gal,
                    bins_tot - 1,
                    bins_tot - 1)
            )
            flux_nonoise_save = []
            for ind_iter in range(mult_iter):
                continuum = (
                        la_e_comp[ind_iter, :, np.newaxis] * one_J_arr[
                                                        ind_iter, :,
                                                        :] * np.exp(
                    -td[ind_iter]) * tau_CGM(
                    Muv[ind_iter]) / (
                                                                       4 * np.pi * Cosmo.luminosity_distance(
                                                                   redshifts_of_mocks[
                                                                       ind_iter]
                                                               ).to(
                                                                   u.cm).value ** 2)[
                                                               :,
                                                               np.newaxis]
                )
                full_flux_res = full_res_flux(continuum,
                                              redshift)
                flux_nonoise_save.append(np.copy(full_flux_res))

                full_flux_res += np.random.normal(
                    0,
                    noise_on_the_spectrum,
                    np.shape(full_flux_res)
                )
                for bin_i, wav_dig_i in zip(
                        range(2, bins_tot - 1), wave_em_dig_arr
                ):
                    flux_spectrum_mock[ind_iter, :, bin_i - 1,
                    :bin_i] = perturb_flux(
                        full_flux_res, bin_i
                    )
                print(flux_spectrum_mock)


        else:
            flux_spectrum_mock = np.zeros(
                (
                    n_gal,
                    bins_tot - 1,
                    bins_tot - 1)
            )
            one_J = one_J_arr[0]
            # print(np.shape(la_e[:, np.newaxis] * one_J[:n_gal,:]), np.shape(td), np.shape(tau_CGM(
            #     Muv)))
            continuum = (
                    la_e_comp[:, np.newaxis] * one_J[:n_gal, :] * np.exp(
                -td) * tau_CGM(
                Muv) / (
                            4 * np.pi * Cosmo.luminosity_distance(
                        7.5
                    ).to(u.cm).value ** 2)
            )

            full_flux_res = full_res_flux(continuum, redshift)
            flux_nonoise_save = np.copy(full_flux_res)

            full_flux_res += np.random.normal(
                0,
                noise_on_the_spectrum,
                np.shape(full_flux_res)
            )
            for bin_i, wav_dig_i in zip(
                    range(2, bins_tot), wave_em_dig_arr
            ):
                flux_spectrum_mock[:, bin_i - 1, :bin_i] = perturb_flux(
                    full_flux_res, bin_i
                )
            print(flux_spectrum_mock)

    cl_load.close_file()

    like_on_flux = flux_spectrum_mock
    bins_likelihood = []
    additive_factors = []
    for bin_i_choice in range(2, bins_tot - 1):
        try:
            list_of_indices = [
                np.where(
                    flux_spectrum_mock[0][i][bin_i_choice - 1][
                    :bin_i_choice] > 3 * noise_on_the_spectrum
                )[0] for i in range(n_gal)
            ]
        except IndexError:
            list_of_indices = [
                np.where(
                    flux_spectrum_mock[i][bin_i_choice - 1][
                    :bin_i_choice] > 3 * noise_on_the_spectrum
                )[0] for i in range(n_gal)
            ]
        if len(np.where(
                np.array(
                    [
                        list(
                            np.concatenate(list_of_indices).ravel()
                        ).count(i) for i in range(bin_i_choice)
                    ]
                ) > 6
        )[0]) == 0:
            print("For some reason, no bins were selected, check this out:",
                  np.array(
                      [
                          list(
                              np.concatenate(list_of_indices).ravel()
                          ).count(i) for i in range(bin_i_choice)
                      ]
                  ))
            print("for bin choice,", bin_i_choice, "and indices",
                  list_of_indices)
            raise ValueError
        bins_likelihood.append(
            np.where(
                np.array(
                    [
                        list(
                            np.concatenate(list_of_indices).ravel()
                        ).count(i) for i in range(bin_i_choice)
                    ]
                ) > 6
            )[0]  # because it's a tuple
        )
        try:
            # additive_factors.append(
            #     10 * np.abs(
            #         np.min(flux_noise_mock[0,:,bin_i_choice-1,:bin_i_choice])
            #     )
            # )
            additive_factors.append(1e-18 * (noise_on_the_spectrum/2e-20))
        except IndexError:
            # additive_factors.append(
            #     10 * np.abs(
            #         np.min(flux_noise_mock[:,bin_i_choice-1,:bin_i_choice])
            #     )
            # ) #5 is probably not enough for the noise since I'm multiplying it by 2.
            additive_factors.append(1e-18 * (noise_on_the_spectrum/2e-20))
    print(
        additive_factors,
        bins_likelihood,
        flush=True
    )

    z_min = -5.0
    z_max = 5.0
    z_grid = np.linspace(z_min, z_max, n_grid)

    x_grid = np.linspace(-5.0, 5.0, n_grid)
    y_grid = np.linspace(-5.0, 5.0, n_grid)
    r_grid = np.linspace(r_min, r_max, n_grid)
    print("Preparing for start", flush=True)

    if mult_iter:
        like_grid_tau_top = np.zeros(
            (
                len(x_grid), len(y_grid), len(z_grid), len(r_grid),
                np.shape(xd)[1],
                mult_iter)
        )
        like_grid_int_top = np.zeros(
            (
                len(x_grid), len(y_grid), len(z_grid), len(r_grid),
                np.shape(xd)[1],
                mult_iter)
        )
        like_grid_spec_top = np.zeros(
            (
                len(x_grid),
                len(y_grid),
                len(z_grid),
                len(r_grid),
                np.shape(xd)[1],
                bins_tot - 1,
                mult_iter
            )
        )
        for ind_iter in range(mult_iter):
            like_calc = Parallel(
                n_jobs=25
            )(
                delayed(
                    _get_likelihood_cache
                )(
                    index,
                    xb,
                    yb,
                    zb,
                    rb,
                    xd[ind_iter],
                    yd[ind_iter],
                    zd[ind_iter],
                    data[ind_iter],
                    n_iter_bub,
                    redshift=redshift,
                    muv=Muv[ind_iter],
                    beta_data=-2.0 * np.ones(np.shape(Muv[ind_iter])),
                    la_e_in=la_e[ind_iter],
                    flux_int=flux_tau[ind_iter],
                    flux_limit=flux_limit,
                    like_on_flux=like_on_flux[ind_iter],
                    n_inside_tau=n_inside_tau,
                    bins_tot=bins_tot,
                    cache=use_cache,
                    constrained_prior=constrained_prior,
                    reds_of_galaxies=redshifts_of_mocks[ind_iter],
                    cache_dir=use_cache,
                    consistent_noise=consistent_noise,
                    noise_on_the_spectrum=noise_on_the_spectrum,
                    additive_factors=additive_factors,
                    bins_likelihood=bins_likelihood,
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
                 np.shape(xd)[1])
            )
            likelihood_grid_int = likelihood_grid_int.reshape(
                (len(x_grid), len(y_grid), len(z_grid), len(r_grid),
                 np.shape(xd)[1])
            )
            likelihood_grid_spec = likelihood_grid_spec.reshape(
                (
                    len(x_grid),
                    len(y_grid),
                    len(z_grid),
                    len(r_grid),
                    np.shape(xd)[1],
                    bins_tot - 1
                )
            )
            like_grid_tau_top[:, :, :, :, :, ind_iter] = likelihood_grid_tau
            like_grid_int_top[:, :, :, :, :, ind_iter] = likelihood_grid_int
            like_grid_spec_top[:, :, :, :, :, :,
                ind_iter] = likelihood_grid_spec
    else:
        like_calc = Parallel(
            n_jobs=25
        )(
            delayed(
                _get_likelihood_cache
            )(
                index,
                xb,
                yb,
                zb,
                rb,
                xd,
                yd,
                zd,
                data,
                n_iter_bub,
                redshift=redshift,
                muv=Muv,
                beta_data = -2.0 * np.ones(np.shape(Muv)),
                la_e_in=la_e,
                flux_int=flux_tau,
                flux_limit=flux_limit,
                like_on_flux=like_on_flux,
                n_inside_tau=n_inside_tau,
                bins_tot=bins_tot,
                cache=use_cache,
                constrained_prior=constrained_prior,
                reds_of_galaxies=redshifts_of_mocks,
                cache_dir=use_cache,
                consistent_noise=consistent_noise,
                noise_on_the_spectrum=noise_on_the_spectrum,
                additive_factors=additive_factors,
                bins_likelihood=bins_likelihood,
            ) for index, (xb, yb, zb, rb) in enumerate(
                itertools.product(x_grid, y_grid, z_grid, r_grid)
            )
        )
        like_calc.sort(key=lambda x: x[0])
        likelihood_grid_tau = np.array([l[1][0] for l in like_calc])
        likelihood_grid_tau = likelihood_grid_tau.reshape(
            (len(x_grid), len(y_grid), len(z_grid), len(r_grid), len(xd))
        )
        likelihood_grid_int = np.array([l[1][1] for l in like_calc])
        likelihood_grid_int = likelihood_grid_int.reshape(
            (len(x_grid), len(y_grid), len(z_grid), len(r_grid), len(xd))
        )
        likelihood_grid_spec = np.array([l[1][2] for l in like_calc])
        likelihood_grid_spec = likelihood_grid_spec.reshape(
            (
                len(x_grid),
                len(y_grid),
                len(z_grid),
                len(r_grid),
                len(xd),
                bins_tot - 1
            )
        )

    dict_to_save_data = dict()
    if mult_iter:
        dict_to_save_data['likelihoods_tau'] = like_grid_tau_top
        dict_to_save_data['likelihoods_int'] = like_grid_int_top
        dict_to_save_data['likelihoods_spec'] = like_grid_spec_top
    else:
        dict_to_save_data['likelihoods_tau'] = likelihood_grid_tau
        dict_to_save_data['likelihoods_int'] = likelihood_grid_int
        dict_to_save_data['likelihoods_spec'] = likelihood_grid_spec

    if gauss_distr:
        dict_to_save_data['la_e_comp'] = la_e_comp
        dict_to_save_data['la_e_orig'] = la_e_orig
        dict_to_save_data['area_factor'] = area_factor
        dict_to_save_data['flux_spectrum'] = flux_spectrum_mock
        dict_to_save_data['flux_integrated'] = flux_tau
    cl_save = HdF5SaveCached(
        n_gal,
        n_iter_bub,
        n_inside_tau,
        save_dir
    )
    cl_save.save_datasets(dict_to_save_data)
    cl_save.close_file()