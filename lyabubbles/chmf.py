"This piece of code includes only the chmf class."

from astropy.cosmology import Planck18 as cosmo
import numpy as np
from scipy import integrate
from astropy import units as u
import os
from joblib import Parallel, delayed
import scipy
import scipy.interpolate
from scipy.special import erfc
from functools import cached_property


class chmf:
    def z_drag_calculate(self):
        z_drag = 0.313*(self.omhh**-0.419) * (1 + 0.607*(self.omhh** 0.674));
        z_drag = 1 + z_drag*(cosmo.Ob0*cosmo.h**2)**(0.238*self.omhh**0.223);
        z_drag *= 1291 * self.omhh**0.251 / (1 + 0.659*self.omhh**0.828);
        return z_drag
    
    def alpha_nu_calculation(self):
        alpha_nu = (self.f_c/self.f_cb) * (2*(self.p_c+self.p_cb)+5)/(4*self.p_cb+5.0)
        alpha_nu *= 1 - 0.553*self.f_nub+0.126*(self.f_nub**3);
        alpha_nu /= 1-0.193*np.sqrt(self.f_nu)+0.169*self.f_nu;
        alpha_nu *= (1+self.y_d)**(self.p_c-self.p_cb);
        alpha_nu *= 1+ (self.p_cb-self.p_c)/2.0 * (1.0+1.0/(4.0*self.p_c+3.0)/(4.0*self.p_cb+7.0))/(1.0+self.y_d)
        return alpha_nu
    
    def MtoR(self,M):
        if (self.FILTER == 0): ##top hat M = (4/3) PI <rho> R^3
            return ((3*M/(4*np.pi*cosmo.Om0*self.critical_density))**(1.0/3.0))    
    
    def RtoM(self, R):
        if (self.FILTER == 0):
            return ((4.0/3.0)*np.pi*R**3*(cosmo.Om0*self.critical_density))
    
    def __init__(self, z, delta_bias, R_bias):

        self.CMperMPC = 3.086e24
        self.Msun = 1.989e33
        self.TINY = 10**-30
        self.Deltac = 1.686
        self.FILTER = 0
        self.T_cmb = cosmo.Tcmb0.value
        self.theta_cmb = self.T_cmb /2.7
        self.critical_density = cosmo.critical_density0.value*self.CMperMPC*self.CMperMPC*self.CMperMPC/self.Msun
        self.z = z
        self.delta_bias = delta_bias
        self.R_bias = R_bias
        self.omhh=cosmo.Om0*(cosmo.h)**2
        self.z_equality = 25000*self.omhh*self.theta_cmb**-4 - 1.0
        self.k_equality = 0.0746*self.omhh/(self.theta_cmb**2)
        self.z_drag = self.z_drag_calculate()
        self.y_d = (1 + self.z_equality) / (1.0 + self.z_drag)
        self.f_nu = cosmo.Onu0 /cosmo.Om0
        self.f_baryon = cosmo.Ob0 / cosmo.Om0
        self.p_c = -(5 - np.sqrt(1 + 24*(1 - self.f_nu-self.f_baryon)))/4.0
        self.p_cb = -(5 - np.sqrt(1 + 24*(1 - self.f_nu)))/4.0
        self.f_c = 1 - self.f_nu - self.f_baryon
        self.f_cb = 1 - self.f_nu
        self.f_nub = self.f_nu+self.f_baryon
        self.alpha_nu = self.alpha_nu_calculation()
        self.R_drag = 31.5 * cosmo.Ob0*cosmo.h**2 * (self.theta_cmb**-4) * 1000 / (1.0 + self.z_drag)
        self.R_equality = 31.5 * cosmo.Ob0*cosmo.h**2 * (self.theta_cmb**-4) * 1000 / (1.0 + self.z_equality)
        self.sound_horizon = 2.0/3.0/self.k_equality * np.sqrt(6.0/self.R_equality) *np.log( (np.sqrt(1+self.R_drag) \
                    + np.sqrt(self.R_drag+self.R_equality)) / (1.0 + np.sqrt(self.R_equality)) )
        self.beta_c = 1.0/(1.0-0.949*self.f_nub)
        self.N_nu = (1.0)
        self.POWER_INDEX = 0.9667
        self.Radius_8 = 8.0/cosmo.h
        self.SIGMA_8 = 0.8159
        self.M_bias = self.RtoM(self.R_bias)
        
        self.TFmdmparams_q = self.theta_cmb**2/self.omhh
        self.TFmdmparams_geff = np.sqrt(self.alpha_nu)
        self.TFmdmparams_TFm = 1.84*self.beta_c*self.TFmdmparams_geff
        self.TFmdmparams_qnu = 3.92 / np.sqrt(self.f_nu/self.N_nu)
        self.TFmdmparams_TFm2 = (1.2*(self.f_nu**0.64)*(self.N_nu**(0.3+0.6*self.f_nu)))
        self._dicke = None
        self._sigma_cell = None
        
    @cached_property
    def dicke(self):
        if self._dicke is None:
            OmegaM_z=cosmo.Om(self.z)
            dick_z = 2.5*OmegaM_z / ( 1.0/70.0 + OmegaM_z*(209-OmegaM_z)/140.0 + pow(OmegaM_z, 4.0/7.0) )
            dick_0 = 2.5*cosmo.Om0 / ( 1.0/70.0 + cosmo.Om0*(209-cosmo.Om0)/140.0 + pow(cosmo.Om0, 4.0/7.0) )
            self._dicke = dick_z / (dick_0 * (1.0+self.z))
        return self._dicke

    def TFmdm(self,k):
        q = k*self.TFmdmparams_q
        gamma_eff=self.TFmdmparams_geff + (1.0-self.TFmdmparams_geff)/(1.0+(0.43*k*self.sound_horizon)** 4)
        q_eff = q/gamma_eff
        TF_m= np.log(np.e + self.TFmdmparams_TFm * q_eff)
        TF_m /= TF_m + q_eff**2 * (14.4 + 325.0/(1.0+60.5*(q_eff**1.11)))
        q_nu = self.TFmdmparams_qnu * q
        TF_m *= 1.0 + self.TFmdmparams_TFm2 /((q_nu**-1.6)+(q_nu**0.8))

        return TF_m

    def dsigma_dk(self, k, R):

        T = self.TFmdm(k)
        p = k**self.POWER_INDEX * T * T
        kR = k*R

        #if ( (kR) < 1.0e-4 ):
        #    w = 1.0
        #else:
        w = 3.0 * (np.sin(kR)/kR**3 - np.cos(kR)/kR**2)
        return k*k*p*w*w         

    def sigma_norm(self,):
        kstart = 1.0*(10**-99)/self.Radius_8
        kend = 350.0/self.Radius_8

        lower_limit = kstart
        upper_limit = kend

        result = integrate.quad(self.dsigma_dk, 0, np.inf, args=(self.Radius_8,), limit=1000, epsabs=10**-20)[0]
        return self.SIGMA_8/np.sqrt(result)

    def sigma_z0(self, M):

        Radius = self.MtoR(M);
        kstart = 1.0*10**(-99)/Radius;
        kend = 350.0/Radius;

        lower_limit = kstart
        upper_limit = kend

        integral = integrate.quad(self.dsigma_dk, kstart, kend, args=(Radius,), limit=1000, epsabs=10**-20)
        result = integral[0]
    
        return self.sigma_normalization * np.sqrt(result)

    def dsigmasq_dm(self, k, R):

        T = self.TFmdm(k);
        p = k**self.POWER_INDEX * T * T;

        kR = k * R;
        #if ( (kR) < 1.0*10**(-4) ):
        #    w = 1.0;
        #else:
        w = 3.0 * (np.sin(kR)/kR**3 - np.cos(kR)/kR**2)


        #if ( (kR) < 1.0*10**(-10) ):
        #    dwdr = 0
        #else:
        dwdr = 9*np.cos(kR)*k/kR**3 + 3*np.sin(kR)*(1 - 3/(kR*kR))/(kR*R)
        drdm = 1.0 / (4.0*np.pi *cosmo.Om0* self.critical_density * R*R);

        return k*k*p*2*w*dwdr*drdm;
    
    def dsigmasqdm_z0(self, M):
        Radius = self.MtoR(M)

        kstart = 1.0e-99/Radius;
        kend = 350.0/Radius;

        lower_limit = kstart
        upper_limit = kend
        result = integrate.quad (self.dsigmasq_dm, kstart, kend, args=(Radius,), limit=1000, epsabs=10**-20)
        return self.sigma_normalization ** 2 * result[0]

    
    
    def dnbiasdM( self, M):
        if ((self.M_bias-M) < self.TINY):
            #print("Mass of the halo bigger than the overdensity mass, not good, stopping now and returing 0")
            return(0)
        delta = self.Deltac/self.dicke - self.delta_bias

        sig_o = self.sigma_z0(self.M_bias);
        sig_one = self.sigma_z0(M);
        sigsq = sig_one*sig_one - sig_o*sig_o
        return -(self.critical_density*cosmo.Om0)/M /np.sqrt(2*np.pi) \
            *delta*((sig_one**2 - sig_o**2)**(-1.5))*(np.e**( -0.5*delta**2/(sig_one**2-sig_o**2))) \
            *self.dsigmasqdm_z0(M)
        
    def prep_for_hmf(self, log10_Mmin = 6, log10_Mmax = 15, dlog10m = 0.01):
        self.log10_Mmin = log10_Mmin
        self.log10_Mmax = log10_Mmax
        self.dlog10m = dlog10m
        self.bins = 10 ** np.arange(self.log10_Mmin, self.log10_Mmax, self.dlog10m)
        self.sigma_z0_array = np.zeros(len(self.bins))
        self.sigma_derivatives = np.zeros(len(self.bins))
        self.sigma_normalization = self.sigma_norm()
        for index, mass in enumerate(self.bins):
            self.sigma_z0_array[index] = self.sigma_z0(mass)
            self.sigma_derivatives[index] = self.dsigmasqdm_z0(mass)
    
    def run_hmf(self, delta_bias):
        self.delta_bias = delta_bias
        delta = self.Deltac/self.dicke - delta_bias
        sigma_array = self.sigma_z0_array**2 - self.sigma_cell**2
        self.hmf = np.zeros(len(self.bins))
        if delta<0:
            print("Something went wrong, cell overdensity bigger than collapse\n")
            print("this cell collapsed already, returning 0")
            return 0
        for index, mass in enumerate(self.bins):
            if mass<self.M_bias:
                self.hmf[index] = -(self.critical_density*cosmo.Om0)/mass/np.sqrt(2*np.pi) * delta * ((sigma_array[index])**(-1.5)) * \
                                (np.e**(-0.5*delta**2/(sigma_array[index]))) * self.sigma_derivatives[index]
            else:
                self.hmf[index] = 0.0
        return self.bins, self.hmf
    
    def dndMnormal( self, M):
        if ((self.M_bias-M) < self.TINY):
            #print("Mass of the halo bigger than the overdensity mass, not good, stopping now and returing 0")
            return(0)
        delta = self.Deltac/self.dicke
        sig_one = self.sigma_z0(M);
        sigsq = sig_one*sig_one
        return -(self.critical_density*cosmo.Om0)/M /np.sqrt(2*np.pi) \
            *delta*((sig_one**2)**(-1.5))*(np.e**( -0.5*delta**2/(sig_one**2))) \
            *self.dsigmasqdm_z0(M)

    @cached_property
    def sigma_cell(self):
        if self._sigma_cell is None:
            self._sigma_cell = self.sigma_z0(self.M_bias)
        return self._sigma_cell
    
#     def run_hmf(self, log10_Mmin = 6, log10_Mmax = 15, dlog10m = 0.01 ):
#         self.log10_Mmin = log10_Mmin
#         self.log10_Mmax = log10_Mmax
#         self.dlog10m = dlog10m
#         self.bins = 10 ** np.arange(self.log10_Mmin, self.log10_Mmax, self.dlog10m)
#         self.hmf = np.zeros(len(self.bins))
#         for i, mass in enumerate(self.bins):
#             self.hmf[i] = self.dnbiasdM(mass * cosmo.h)
#         return (self.bins, self.hmf)
    
    def run_hmf_normal(self, log10_Mmin = 6, log10_Mmax = 15, dlog10m = 0.01 ):
        self.log10_Mmin_normal = log10_Mmin
        self.log10_Mmax_normal = log10_Mmax
        self.dlog10m_normal = dlog10m
        self.bins_normal = 10 ** np.arange(self.log10_Mmin_normal, self.log10_Mmax_normal, self.dlog10m_normal)
        self.hmf_normal = np.zeros(len(self.bins_normal))
        for i, mass in enumerate(self.bins_normal):
            self.hmf_normal[i] = self.dndMnormal(mass)
        return (self.bins_normal, self.hmf_normal)
    
    def cumulative_number(self):
        dndlnm = self.bins * self.hmf
        if self.bins[-1] < self.bins[0] * 10 ** 18 / self.bins[3]:
            m_upper = np.arange(
                np.log(self.bins[-1]), np.log(10 ** 18), np.log(self.bins[1]) - np.log(self.bins[0])
            )
            mf_func = _spline(np.log(self.bins), np.log(dndlnm), k=1)
            mf = mf_func(m_upper)

            int_upper = integrate.simps(np.exp(mf), dx=m_upper[2] - m_upper[1], even="first")
        else:
            int_upper = 0

        # Calculate the cumulative integral (backwards) of [m*]dndlnm
        self.cumnum = np.concatenate(
            (
                integrate.cumtrapz(dndlnm[::-1], dx=np.log(self.bins[1]) - np.log(self.bins[0]))[::-1],
                np.zeros(1),
            )
        )
        self.cumnum+=int_upper
        return self.cumnum
    
    def cumulative_mass(self):
        dndlnm = self.bins * self.hmf

        if self.bins[-1] < self.bins[0] * 10 ** 18 / self.bins[3]:
            m_upper = np.arange(
                np.log(self.bins[-1]), np.log(10 ** 18), np.log(self.bins[1]) - np.log(self.bins[0])
            )
            mf_func = _spline(np.log(self.bins), np.log(dndlnm), k=1)
            mf = mf_func(m_upper)
            int_upper = integrate.simps(
                np.exp(m_upper + mf), dx=m_upper[2] - m_upper[1], even="first"
            )
        else:
            int_upper = 0

        self.cummass = np.concatenate(
            (
                integrate.cumtrapz(self.bins[::-1] * dndlnm[::-1], dx=np.log(self.bins[1]) - np.log(self.bins[0]))[
                    ::-1
                ],
                np.zeros(1),
            )
        )
        self.cummass += int_upper
        return self.cummass
    
    def f_coll_calc(self, M, remove_delta=False): #collapsed fraction of halos above mass M
        if not remove_delta:
            delta = self.Deltac/self.dicke - self.delta_bias
            sigma_bias = np.sqrt(self.sigma_z0(M)**2 - self.sigma_cell**2)
        else:
            delta = self.Deltac/self.dicke
            sigma_bias = self.sigma_z0(M)
        q = abs(delta/np.sqrt(2)/sigma_bias)
        t = 1.0/(1.0+0.5*q)
        self.f_coll = t*np.exp(-q**2 - 1.2655122+t*(1.0000237+t*(0.374092+t*(0.0967842+t*(-0.1862881+t*(0.2788681+t*(-1.13520398+t*(1.4885159+t*(-0.82215223+t*0.17087277)))))))))
        return self.f_coll
        
    def mass_coll_grt(self):
        return self.f_coll_calc(10**self.log10_Mmin) * 4*np.pi/3*self.R_bias**3 * self.critical_density #* (1+self.delta_bias)
    
    def dNdM_st(self, M):
        sigma = self.sigma_z0(M) * self.dicke
        dsigmadm = self.dsigmasqdm_z0(M) * self.dicke**2/(2*sigma)
        self.SHETH_a = 0.73
        self.SHETH_A = 0.353
        self.SHETH_p = 0.175
        nuhat = np.sqrt(self.SHETH_a) * self.Deltac/sigma
        return -(self.critical_density*cosmo.Om0)/M * (dsigmadm/sigma) * np.sqrt(2/np.pi) * self.SHETH_A * (1+nuhat**(-2*self.SHETH_p)) * nuhat * np.exp(-nuhat**2/2)
    
    def dNbiasdM_st(self, M):
        delta = self.Deltac/self.dicke - self.delta_bias
        sig_o = self.sigma_z0(self.M_bias);
        sigma = self.sigma_z0(M)
        sigsq = sigma*sigma - sig_o*sig_o
        dsigmadm = self.dsigmasqdm_z0(M) /(2*np.sqrt(sigsq))
        self.SHETH_a = 0.73
        self.SHETH_A = 0.353
        self.SHETH_p = 0.175
        nuhat = np.sqrt(self.SHETH_a) * delta/np.sqrt(sigsq)
        return -(self.critical_density*cosmo.Om0)/M * (dsigmadm/np.sqrt(sigsq)) * np.sqrt(2/np.pi) * self.SHETH_A * (1+nuhat**(-2*self.SHETH_p)) * nuhat * np.exp(-nuhat**2/2)
    
    def prep_for_hmf_st(self, log10_Mmin = 6, log10_Mmax = 15, dlog10m = 0.01):
        self.log10_Mmin = log10_Mmin
        self.log10_Mmax = log10_Mmax
        self.dlog10m = dlog10m
        self.bins = 10 ** np.arange(self.log10_Mmin, self.log10_Mmax, self.dlog10m)
        self.sigma_z0_array_st = np.zeros(len(self.bins))
        self.sigma_derivatives_st = np.zeros(len(self.bins))
        self.sigma_normalization = self.sigma_norm()

        for index, mass in enumerate(self.bins):
            self.sigma_z0_array_st[index] = self.sigma_z0(mass)
            self.sigma_derivatives_st[index] = self.dsigmasqdm_z0(mass)
    
    def run_hmf_st(self, delta_bias):
        self.delta_bias = delta_bias
        delta = self.Deltac/self.dicke - delta_bias
        sigma_array = self.sigma_z0_array_st**2 - self.sigma_cell**2
        self.hmf_st = np.zeros(len(self.bins))
        self.SHETH_a = 0.73
        self.SHETH_A = 0.353
        self.SHETH_p = 0.175
        dsigmadm = self.sigma_derivatives_st / (2*np.sqrt(sigma_array))
        nuhat = np.sqrt(self.SHETH_a) * delta/np.sqrt(sigma_array)
        if delta<0:
            print("Something went wrong, cell overdensity bigger than collapse\n")
            print("this cell collapsed already, returning 0")
            return 0
        
        for index, mass in enumerate(self.bins):
            if mass<self.M_bias:
                self.hmf_st[index] = -(self.critical_density*cosmo.Om0)/mass/np.sqrt(2/np.pi) *(dsigmadm[index]/np.sqrt(sigma_array[index]))* (nuhat[index])*  \
                                self.SHETH_A *(1+(nuhat[index])**(-2*self.SHETH_p)) *(np.e**(-0.5*(nuhat[index])**2))
            else:
                self.hmf_st[index] = 0.0
        return self.bins, self.hmf_st

    def dfdM_st(self, logM):
        M = np.exp(logM)
        MassFunction = self.dNdM_st(M)
        return MassFunction * M * M

    def f_coll_st(self,M):
        f_coll_st = integrate.quad (self.dfdM_st, np.log(M), np.log(10**19), limit=1000, epsabs=10**-10)
        return f_coll_st[0] / (cosmo.Om0*self.critical_density)
    
    def mass_coll_grt_ST(self, delta_bias, mass = None):
        """
            Modified collapsed fraction to be in line with the modified hmf.

            Paramereters
            ------------
            delta_bias: float,
                overdensity to calculate the collapsed fraction for.
            mass: float, optional;
                minimum mass for which we calculate the collapsed fraction.
                If not given, minimum hmf mass will be used.
                Given in log-scale.
        """
        if not mass:
            mass = self.log10_Mmin
            sigma_currmass = self.sigma_z0_array_st[0]
        else:
            sigma_currmass = np.interp(10**mass, self.bins, self.sigma_z0_array_st)
        fraction =  self.f_coll_st(10**mass) / self.f_coll_calc(10**mass)
        s = np.sqrt(2 * (sigma_currmass**2 - self.sigma_cell**2))
        er_f = erfc ((self.Deltac/self.dicke - delta_bias)/ s)
        return fraction* er_f * 4*np.pi/3*self.R_bias**3 * self.critical_density * cosmo.Om0
    
    def prep_collapsed_fractions(self, check_cache=True):
        if check_cache and os.path.exists('/home/inikolic/projects/stochasticity/_cache/derivatives_{}_{}_{:.5f}.txt'.format(self.log10_Mmin, self.log10_Mmax, self.dlog10m)):
            print("Managed to read the sigma files")
            self.derivative_ratios = np.loadtxt('/home/inikolic/projects/stochasticity/_cache/derivatives_{}_{}_{:.5f}.txt'.format(self.log10_Mmin, self.log10_Mmax, self.dlog10m))
            self.collapsed_ratios = np.loadtxt('/home/inikolic/projects/stochasticity/_cache/ratios_{}_{}_{:.5f}.txt'.format(self.log10_Mmin, self.log10_Mmax, self.dlog10m))
        else:
            print("Didn't manage to read the sigma files.")
            bins_for_deriv = 10**np.arange(self.log10_Mmin-2 * self.dlog10m, self.log10_Mmax + 2*self.dlog10m, self.dlog10m)
        #this way I have enough derivatives
            remember_index= len(bins_for_deriv)
            collapsed_ratios_full = np.zeros(int((self.log10_Mmax- self.log10_Mmin)/self.dlog10m + 4))
        #for index, mass in enumerate(bins_for_deriv):
        #    collapsed_ratios_full[index] = self.f_coll_st(mass) / self.f_coll_calc(mass)
            collapsed_ratios_full = Parallel(n_jobs=50)(delayed(self.f_coll_st)(mass) for mass in bins_for_deriv)
            for i in range(len(bins_for_deriv)):
                #collapsed_denominator = self.f_coll_calc(bins_for_deriv[i])
                if bins_for_deriv[i]<self.M_bias and self.f_coll_calc(bins_for_deriv[i], remove_delta=True)>0.0:
                    collapsed_ratios_full[i]/=self.f_coll_calc(bins_for_deriv[i], remove_delta=True)
                else:
                    if not remember_index:
                        remember_index = i
                    collapsed_ratios_full[i]=0.0
            collapsed_ratios_spline = scipy.interpolate.InterpolatedUnivariateSpline(bins_for_deriv[:remember_index], collapsed_ratios_full[:remember_index])
            derivative_spline = collapsed_ratios_spline.derivative()
            self.derivative_ratios = derivative_spline(bins_for_deriv[2:remember_index])
            self.collapsed_ratios = np.array(collapsed_ratios_full[2:-2])

            np.savetxt('/home/inikolic/projects/stochasticity/_cache/derivatives_{}_{}_{:.5f}.txt'.format(self.log10_Mmin, self.log10_Mmax, self.dlog10m), self.derivative_ratios)
            np.savetxt('/home/inikolic/projects/stochasticity/_cache/ratios_{}_{}_{:.5f}.txt'.format(self.log10_Mmin, self.log10_Mmax, self.dlog10m), self.collapsed_ratios)
        return self.derivative_ratios

    def ST_hmf(self, delta_inst):
        #self.hmf_ST = np.zeros((len(self.bins)))
        delta = self.Deltac/self.dicke - delta_inst
        sigma_array = self.sigma_z0_array_st**2 - self.sigma_cell**2
        #print("Is hmf the problem?")
        #print("collapsedratios", self.collapsed_ratios[:10], type(self.collapsed_ratios),self.derivative_ratios[:10], type(self.derivative_ratios),self.bins[:10], type(self.bins))
        #print("sigma_array", sigma_array[:10], type(sigma_array))
        self.hmf_ST = self.collapsed_ratios * -(self.critical_density*cosmo.Om0)/self.bins/np.sqrt(2*np.pi) * delta * ((sigma_array)**(-1.5)) * \
                                (np.e**(-0.5*delta**2/(sigma_array))) * self.sigma_derivatives_st + self.critical_density*cosmo.Om0 / self.bins * self.derivative_ratios[1:-1] * scipy.special.erfc(delta/(np.sqrt(2*sigma_array)))
        #print("I guess not!")
        #for index,mass in enumerate(self.bins):
        #    if mass<self.M_bias and index<len(self.derivative_ratios):
        #        biased_hmf_part = -(self.critical_density*cosmo.Om0)/self.bins[index]/np.sqrt(2*np.pi) * delta * ((sigma_array[index])**(-1.5)) * \
        #                        (np.e**(-0.5*delta**2/(sigma_array[index]))) * self.sigma_derivatives_st[index]
        #        self.hmf_ST[index] = self.collapsed_ratios[index] * biased_hmf_part

        #        self.hmf_ST[index]+= self.critical_density*cosmo.Om0 / self.bins[index] * self.derivative_ratios[index] * scipy.special.erfc(delta/(np.sqrt(2*sigma_array[index])))
        #    else:
        #        self.hmf_ST[index]=0.0
        return self.hmf_ST
