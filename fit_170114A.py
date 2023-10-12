import time
import os
import numpy as np

import pymultinest
from threeML import *


################ Dowloading POLAR data/response ################
POLAR_data_path = "https://www.astro.unige.ch/polar/sites/default/Public/photons/"

if not os.path.exists("polarization_170114A_rsp.h5"):
    os.system("wget {}polarization_170114A_rsp.h5".format(POLAR_data_path))  # polarization response
    
if not os.path.exists("polar_170114A_rsp.h5"):
    os.system("wget {}polar_170114A_rsp.h5".format(POLAR_data_path))  # data + spectral response


trigger_time = 1484431269.5000

################ POLAR spectral data ################

polar = TimeSeriesBuilder.from_polar_spectrum(name='polar_spec',
                                               polar_hdf5_file='polar_170114A_rsp.h5',
                                               trigger_time=trigger_time)

polar.set_active_time_interval('-0.2-8.9')
polar.set_background_interval('-35--10','25-75')
polar.create_time_bins(start=-30.0, stop=30.0, method='constant', dt=0.1)

polar_spec = polar.to_spectrumlike()
polar_spec.set_active_measurements('30-750')

polar_spec.use_effective_area_correction(0.7, 1.3)


################ POLAR polarization data ################

polar_polarization_ts = TimeSeriesBuilder.from_polar_polarization(name='polar_pol',
                                               polar_hdf5_file='polar_170114A_rsp.h5',
                                               polar_hdf5_response='polarization_170114A_rsp.h5',          
                                               trigger_time=trigger_time)

polar_polarization_ts.set_background_interval('-35--10','25-75')
polar_polarization_ts.set_active_time_interval('-0.2-8.9')
polar_data = polar_polarization_ts.to_polarlike()

polar_data.use_effective_area_correction(0.7, 1.5)


################ GBM data ################

gbm_cat = FermiGBMBurstCatalog()

gbm_cat.query_sources('GRB170114917')

dl = download_GBM_trigger_data('bn170114917',detectors=['n1','n5','n8','b0'])

n1 = TimeSeriesBuilder.from_gbm_tte('n1',dl['n1']['tte'],dl['n1']['rsp'],verbose=False)
n5 = TimeSeriesBuilder.from_gbm_tte('n5',dl['n5']['tte'],dl['n5']['rsp'],verbose=False)
n8 = TimeSeriesBuilder.from_gbm_tte('n8',dl['n8']['tte'],dl['n8']['rsp'],verbose=False)
b0 = TimeSeriesBuilder.from_gbm_tte('b0',dl['b0']['tte'],dl['b0']['rsp'],verbose=False)


n1.set_background_interval('-50--5','15-100')
n1.set_active_time_interval('3.6-4.8')
n1.view_lightcurve(-50,100)

n5.set_background_interval('-50--5','15-100')
n5.set_active_time_interval('3.6-4.8')
n5.view_lightcurve(-50,100)

n8.set_background_interval('-50--5','15-100')
n8.set_active_time_interval('3.6-4.8')
n8.view_lightcurve(-50,100)

b0.set_background_interval('-50--5','15-100')
b0.set_active_time_interval('3.6-4.8')
b0.view_lightcurve(-50,100)


n1_spec = n1.to_spectrumlike()
n1_spec.set_active_measurements('8.1-510')
n5_spec = n5.to_spectrumlike()
n5_spec.set_active_measurements('8.1-510')
n8_spec = n8.to_spectrumlike()
n8_spec.set_active_measurements('8.1-510')
b0_spec = b0.to_spectrumlike()
b0_spec.set_active_measurements('300-35000')



################ Spectral model ################

band = Band()

band.xp.prior = Log_normal(mu=np.log(300),sigma=np.log(100))
band.xp.bounds = (None, None)

band.K.bounds = (1E-10, None)
band.K.prior = Log_uniform_prior(lower_bound=1E-5, upper_bound=1E2)

band.alpha.bounds = (-1.5, 1.0)
band.alpha.prior = Truncated_gaussian(mu=-1, sigma=0.5, lower_bound=-1.5, upper_bound=1.0)

band.beta.bounds = (None, -1.5)
band.beta.prior = Truncated_gaussian(mu=-3.,sigma=0.6, lower_bound=-5, upper_bound=-1.5)



################ Polarization model ################

lp = LinearPolarization(10,10)
lp.angle.set_uninformative_prior(Uniform_prior)
lp.degree.prior = Uniform_prior(lower_bound=0.1, upper_bound=100.0)
#lp.degree.prior = Log_uniform_prior(lower_bound=1E-10,
 #                                       upper_bound=100)
#lp.degree.prior = Truncated_gaussian(mu=20,sigma=30, lower_bound=0, upper_bound=100)
lp.degree.value=0
lp.angle.value=10





sc = SpectralComponent('synch', band, lp)

ps = PointSource('polar_GRB',0,0,components=[sc])

model = Model(ps)
datalist = DataList(polar_data, polar_spec, n1_spec, n5_spec, n8_spec, b0_spec)



# BAYES

bayes = BayesianAnalysis(model,datalist)
bayes.set_sampler("multinest")
wrapped = [0] * len(model.free_parameters)
wrapped[5] = 1

bayes.sampler.setup(n_live_points=100,
                           resume = False,
                           importance_nested_sampling=False,
                           verbose=True,
                           wrapped_params=wrapped,
                           chain_name='chains/synch_p2')

bayes.sample()


################ Save results ################

bayes.results.write_to("fit_results.fits", overwrite=True)


################ Plot spectrum ################

bayes.restore_median_fit()
fig = display_spectrum_model_counts(bayes)
fig.savefig("count_spec.pdf", bbox_inches="tight")


################ Play with the results ################

# ar = load_analysis_results(results_path + fits_file)
