from astropy import constants as const
from astropy import units as u
import numpy as np
from astropy.cosmology import Planck15
from astropy.cosmology import Planck18 as Cosmo
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
import scipy.integrate as intg
import scipy

wave_Lya = 1215.67 * u.Angstrom
sigma_ion0 = 6.304e-18*u.cm**2.
freq_Lya = (const.c / wave_Lya).to(u.Hz)
wave_em = np.linspace(1214, 1230., 100) * u.Angstrom

def I(x):
    return x**4.5 / (1-x) + 9/7 * x**3.5  + 9/5 * x**2.5 + 3 * x**1.5 + 9 * x**0.5 - np.log(abs((1+x**0.5)/(1-x**0.5)))


def comoving_distance_from_source_Mpc(z_2, z_1):
    """
    COMOVING distance between z_1 and z_2
    """
    R_com = (z_1 - z_2)*(const.c / Planck15.H(z=z_1)).to(u.Mpc)
    return R_com


def wave_to_dv(
        wave
):
    """
    Wavelength to velocity offset.
    :param wave: float;
        wavelength of interest.
    :return dv: float;
        velocity offset.
    """
    return ((wave - wave_Lya)*const.c/wave_Lya).to(u.km/u.s)


def gaussian(
        x,
        mu,
        sig
):
    """
    Gaussian shape at a given position.

    :param x: float;
        x-coordinate for which we evaluate the Gaussian.
    :param mu: float;
        mean of the gaussian.
    :param sig: float;
        STD of the gaussian.

    :return PDF: float,
         Value of the gaussian.
    """
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def dt_dz_func(z):
    return 1. / ((1. + z) * Planck15.H(z))


class LyaCrossSection(object):
    """
    Make Lya cross-section given temperature
    """

    def __init__(self, t=1.e4 * u.K):
        """

        """
        self.t = t

        # Calculate thermal velocity
        self.v_therm = self.v_thermal()
        self.d_freq_Lya = (freq_Lya * self.v_therm / const.c).to(u.Hz)

        # Voigt parameter
        self.av_T = self.av()

        # Cross-section peak
        self.sig_Lya0 = 5.9e-14 * (self.t / 1.e4 / u.K) ** -0.5

        return

    def av(self):
        return 4.7e-4 * (self.t.value / 1.e4) ** (-0.5)

    def voigt_badapprox(self, x):
        """Clumsy approximation for Voigt
        """
        phix = np.exp(-x ** 2.) + self.av_T / np.sqrt(np.pi) / x ** 2.
        phix[phix > 1.] = 1.
        return phix

    def voigt(self, x):
        """Voigt function approximation from Tasitsiomi 2006

        https://ui.adsabs.harvard.edu/abs/2006ApJ...645..792T/abstract

        Good to >1% for T>2K. Correctly normalized to return H(av, x).
        int(phix)  = 1
        int(Voigt) = sqrt(pi)

        Args:
            x (ndarray): dimensionless frequency

        Returns:
            Voigt function
        """

        z = (x ** 2. - 0.855) / (x ** 2. + 3.42)

        q = z * (1 + 21. / x ** 2.) * self.av_T / np.pi / (x ** 2. + 1) * \
            (0.1117 + z * (4.421 + z * (-9.207 + 5.674 * z)))

        phix = np.exp(-x ** 2.) / 1.77245385

        phix[z > 0] += q[z > 0]

        return phix * np.sqrt(np.pi)

    def v_thermal(self):
        """
        Thermal velocity (Maxwell-Boltzmann)
        """
        return (1. / np.sqrt(const.m_p / 2. / const.k_B / self.t)).to(
            u.km / u.s)

    def lya_wave_to_x(self, wave):
        """
        Convert wavelength to x
        """
        freq = (const.c / wave).to(u.Hz)

        # Dimensionless frequency
        x = (freq - freq_Lya) / self.d_freq_Lya

        return x

    def lya_x_to_wave(self, x):
        """
        Convert x to wavelength
        """
        # Wavelength
        wave = const.c / (x * self.d_freq_Lya + freq_Lya)

        return wave.to(u.Angstrom)

    def lya_cross_sec(self, wave, return_x=False):

        x = self.lya_wave_to_x(wave)

        sig_lya = self.sig_Lya0 * self.voigt(x) * u.cm ** 2.

        if return_x:
            return x, sig_lya
        else:
            return sig_lya

    def lya_cross_sec_x(self, x):

        sig_lya = self.sig_Lya0 * self.voigt(x) * u.cm ** 2.

        return sig_lya

    @staticmethod
    def optical_depth_weinberg97(self, z, n_hi):
        """
        Lya optical depth from https://ui.adsabs.harvard.edu/abs/1997ApJ...490..564W/abstract Eqn 3
        for uniform IGM
        tau = πe^2/(m_e c) * f_alpha lambda_alpha * n_HI/H(z)
        """
        tau = 1.34e-17 * u.cm ** 3 / u.s * Planck15.H(z).to(1. / u.s) * n_hi

        return tau.value


def n_h(z, x_p=0.75):
    """IGM hydrogen proper number density
    Args:
        z (ndarray): redshift
        x_p : Helium fraction
    Returns:
        number density in cm^-3
    Cen & Haiman 2000::
    >>> 8.5e-5 * ((1. + z)/8.)**3. / (u.cm**3.)
    """
    return (x_p * Planck15.Ob0 * Planck15.critical_density0 * (1+z)**3.
            / const.m_p).to(u.cm**(-3.))


def alpha_rec_b(t):
    """Recombination rate for `case B` recombination of hydrogen.
       Fitting formulae from Hui & Gnedin (1997) (in the Appendix)
       accurate to ~0.7% from 1 - 10^9K
        input T unit-less (but in K)
       Returns recombination rate in cm**3/s
    """
    lhi = 2 * 157807.0 / t
    alpha_b = 2.753e-14 * lhi**1.5 / (1.0 + (lhi / 2.740)**0.407)**2.242

    return alpha_b * u.cm**3. / u.s


def xhi_r(r, z_s, n_dot_ion, f_esc=1., c=3., delta=1, t=1e4, alpha=-1.8):
    """
    r is proper distance
    Neutral fraction from source
    (Mesinger+04)
    for f_esc = 0.5, N_ion = 1e-57/s
    Args:
        r (float): radius of the bubble.
        z_s: redshift of the source.
        n_dot_ion: ionizing photon production
        f_esc: escape fraction
        c: clumping factor
        delta: overDensity
        t: temperature of IGM
        alpha: gamma scaling.
    """
    j_source = f_esc * n_dot_ion * (alpha/(alpha - 3)) * sigma_ion0

    gamma12_source = 1./(4. * np.pi * r**2.) * j_source
    gamma12_background = 0.  # J_bg * bubbles.Gamma12(z_s) / u.s

    xhi = c * delta * n_h(z_s) * alpha_rec_b(t)/(
            gamma12_background + gamma12_source
    )

    return xhi.to(u.s/u.s)


def optical_depth(
        wave_em,
        t,
        z_min,
        z_max,
        z_s=7.,
        z_bubble_center=None,
        inside_hii=True,
        c_hii=3.,
        xtab_len=100,
        over_density=1.,
        n_dot_ion=1.e57/u.s
):
    """
    Lya optical depth as a function of wavelength
    using definition of optical depth and Lya cross-section
    """
    cross_sec = LyaCrossSection(t)

    # Redshift array
    z_tab_ends = np.array([z_min, z_max])

    # Observed wavelength
    wave_obs = wave_em * (1. + z_s)
    if z_bubble_center is None:
        z_bubble_center = z_s

    # Range of redshifted wavelength and x
    wave_z_ends = wave_obs[:, None]/(1+z_tab_ends)
    x_z_ends = cross_sec.lya_wave_to_x(wave_z_ends).value

    tau = np.zeros(len(wave_obs))
    for ww, w_obs in enumerate(wave_obs):

        # Make xtab
        if (x_z_ends[ww] < 0).all():
            xtab = -np.logspace(
                np.log10(-x_z_ends[ww].min()),
                np.log10(-x_z_ends[ww].max()),
                xtab_len
            )
            xtab = np.sort(xtab)
        elif (x_z_ends[ww] > 0).all():
            xtab = np.logspace(
                np.log10(x_z_ends[ww].min()),
                np.log10(x_z_ends[ww].max()),
                xtab_len)
            xtab = np.sort(xtab)
        else:
            xtab_neg = -np.logspace(
                -1,
                np.log10(-x_z_ends[ww].min()),
                int(xtab_len/2))
            xtab_pos = np.logspace(
                -1,
                np.log10(x_z_ends[ww].max()),
                int(xtab_len/2)
            )
            xtab = np.sort(np.concatenate((xtab_neg, xtab_pos)))

        # Get wave_redshift
        wave_redshift = cross_sec.lya_x_to_wave(xtab)

        # Get z tab
        z_tab = w_obs/wave_redshift - 1.

        # Residual neutral fraction
        if inside_hii:
            r_com = comoving_distance_from_source_Mpc(z_tab, z_bubble_center)
            r_p = r_com / (1+z_bubble_center)
            xhi = xhi_r(
                r=r_p,
                z_s=z_bubble_center,
                n_dot_ion=n_dot_ion,
                f_esc=1.,
                c=c_hii,
                t=t.value
            )
        else:
            xhi = 1.

        # Cross-section
        lya_cross = cross_sec.lya_cross_sec_x(xtab)
        # Calculate optical depth
        pre_fact = (const.c * dt_dz_func(z_tab) * xhi * over_density
                    * n_h(z_tab)).to(1./u.cm**2.)
        d_tau = pre_fact * lya_cross

        tau[ww] = np.trapz(d_tau, z_tab)

    return tau


def z_at_proper_distance(R_p, z_1=7.):
    R_H = (const.c / Cosmo.H(z=z_1)).to(u.Mpc)
    R_com = R_p * (1+z_1)
    return z_1 - R_com/R_H

def hmf_integral_gtm(M, dndm, mass_density=False):
    """
    Cumulatively integrate dn/dm.
    Parameters
    ----------
    M : array_like
        Array of masses.
    dndm : array_like
        Array of dn/dm (corresponding to M)
    mass_density : bool, `False`
        Whether to calculate mass density (or number density).
    Returns
    -------
    ngtm : array_like
        Cumulative integral of dndm.
    Examples
    --------
    Using a simple power-law mass function:
    >>> import numpy as np
    >>> m = np.logspace(10,18,500)
    >>> dndm = m**-2
    >>> ngtm = hmf_integral_gtm(m,dndm)
    >>> np.allclose(ngtm,1/m) #1/m is the analytic integral to infinity.
    True
    The function always integrates to m=1e18, and extrapolates with a spline
    if data not provided:
    >>> m = np.logspace(10,12,500)
    >>> dndm = m**-2
    >>> ngtm =     >>> np.allclose(ngtm,1/m) #1/m is the analytic integral to infinity.
    True
    """
    # Eliminate NaN's
    m = M[np.logical_not(np.isnan(dndm))]
    dndm = dndm[np.logical_not(np.isnan(dndm))]
    dndlnm = m * dndm
    #print(m, dndm, dndlnm)
    if len(m) < 4:
        raise NaNException(
            "There are too few real numbers in dndm: len(dndm) = %s, #NaN's = %s"
            % (len(M), len(M) - len(dndm))
        )

    # Calculate the mass function (and its integral) from the highest M up to 10**18
    if m[-1] < m[0] * 10**18 / m[3]:
        m_upper = np.arange(
            np.log(m[-1]), np.log(10**18), np.log(m[1]) - np.log(m[0])
        )
        mf_func = _spline(np.log(m), np.log(dndlnm), k=1)
        mf = mf_func(m_upper)

        if not mass_density:
            int_upper = intg.simps(np.exp(mf), dx=m_upper[2] - m_upper[1], even="first")
        else:
            int_upper = intg.simps(
                np.exp(m_upper + mf), dx=m_upper[2] - m_upper[1], even="first"
            )
    else:
        int_upper = 0

    # Calculate the cumulative integral (backwards) of [m*]dndlnm
    if not mass_density:
        ngtm = np.concatenate(
            (
                intg.cumtrapz(dndlnm[::-1], dx=np.log(m[1]) - np.log(m[0]))[::-1],
                np.zeros(1),
            )
        )
    else:
        ngtm = np.concatenate(
            (
                intg.cumtrapz(m[::-1] * dndlnm[::-1], dx=np.log(m[1]) - np.log(m[0]))[
                    ::-1
                ],
                np.zeros(1),
            )
        )

    return ngtm# + int_upper

def full_res_flux(
        continuum,
        redshift=7.5
):
    """
    :param continuum:
        continuum flux for this run (it could contain multiple galaxies):
        shape of it will be ((dimensions of gal), 100)
    :return: flux_full_res:
        spectrum at the full resolution corresponding to R~2700 (NirSpec)
    """

    spec_res = wave_Lya.value * (1 + redshift) / 2700
    bins = np.arange(wave_em.value[0] * (1 + redshift),
                     wave_em.value[-1] * (1 + redshift), spec_res)
    wave_em_dig = np.digitize(wave_em.value * (1 + redshift), bins)
    bins_po = np.append(bins, bins[-1] + spec_res)

    #first let's check the shape
    shap = np.shape(continuum)
    flux_full_res = np.zeros((np.product(shap[:-1]), len(bins)))
    for ind in range(np.product(shap[:-1])):
        flux_full_res[ind] = [
            np.trapz(
                x=wave_em.value[wave_em_dig == i + 1],
                y=(
                    continuum.reshape(
                        np.product(shap[:-1]),100
                    )[ind][wave_em_dig == i + 1])
            )
            for i in range(len(bins))
        ]
    return flux_full_res.reshape((*list(shap[:-1]),-1))

def perturb_flux(
        full_res,
        n_bins,
        gaussian_filter=False,
        redshift=7.5,
):
    """

    :param flux_full_res:
        flux at full resolution.
    :param n_bins:
        number of bins to perturb the flux
    :return:
        flux_at_n
    """

    spec_res = wave_Lya.value * (1 + redshift) / 2700
    bins = np.arange(wave_em.value[0] * (1 + redshift),
                     wave_em.value[-1] * (1 + redshift), spec_res)

    bins_rebin = np.linspace(
        wave_em.value[0] * (1 + 7.5),
        wave_em.value[-1] * (1 + 7.5),
        n_bins + 1
    )
    wave_em_dig_rebin = np.digitize(bins, bins_rebin)
    full_res_shape = np.shape(full_res)
    full_res_resh = full_res.reshape((np.product(full_res_shape[:-1]), -1))
    if gaussian_filter:
        flux_0 = np.array(
            [scipy.ndimage.gaussian_filter(np.array(full_res_resh[i]), 1) for i in
             range(len(full_res_resh))]
        )
    else:
        flux_0 = full_res_resh
    flux_rebin = np.array([np.sum(flux_0[:, wave_em_dig_rebin == j + 1], axis=1) for j in
                  range(n_bins)])
    return flux_rebin.T.reshape((*list(full_res_shape[:-1]), n_bins))

def comoving_distance_from_source_Mpc(z_2, z_1):
    """
    COMOVING distance between z_1 and z_2. z_1 > z_2
    """
    R_com = (z_1 - z_2)*(const.c / Planck15.H(z=z_1)).to(u.Mpc)
    return R_com.value