#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import netCDF4 as nc
import numpy as np
import iris.coords as coords
import cf_units
import iris.cube as icube
import pdb
from scipy.stats import kurtosis
from scipy import stats
import seaborn as sns
import scipy.stats as stats
import random
import matplotlib.pyplot as plt
import math
import numpy.ma as ma
import datetime as dt
import iris.plot as iplt
import matplotlib as mpl
import iris

def mean_bias_correction(obs, model):
    """
    Perform mean bias correction on the model data.
    
    Parameters:
    obs (np.array): Observed values, shape (n_months,)
    model (np.array): Model forecast values, shape (n_ensembles, n_months)
    
    Returns:
    np.array: Bias-corrected model values, same shape as input model
    """
    # Calculate the mean across all ensembles and months
    model_mean = np.mean(model)
    obs_mean = np.mean(obs)
    
    # Calculate the correction factor
    correction_factor = obs_mean - model_mean
    
    # Apply the correction factor to the model data
    model_biascor = model + correction_factor
    
    return model_biascor




def ratio_correction(obs, model):
    """
    Perform mean bias correction on the model data.
    
    Parameters:
    obs (np.array): Observed values, shape (n_months,)
    model (np.array): Model forecast values, shape (n_ensembles, n_months)
    
    Returns:
    np.array: Bias-corrected model values, same shape as input model
    """
    # Calculate the mean across all ensembles and months
    model_mean = np.mean(model)
    obs_mean = np.mean(obs)
    model_std = np.std(model)
    obs_std = np.std(obs)
    # Calculate the correction factor
    #correction_factor = model_mean - obs_mean
    sigma= model_std/obs_std
    mu= model_mean - obs_mean * sigma

    # Apply the correction factor to the model data
    model_biascor = (model - mu)/sigma
    return model_biascor


def mean_correction(obs, model):
    """
    Perform mean bias correction on the model data.
    
    Parameters:
    obs (np.array): Observed values, shape (n_months,)
    model (np.array): Model forecast values, shape (n_ensembles, n_months)
    
    Returns:
    np.array: Bias-corrected model values, same shape as input model
    """
    # Calculate the mean across all ensembles and months
    model_mean = np.mean(model)
    obs_mean = np.mean(obs)
    model_std = np.std(model)
    obs_std = np.std(obs)
    # Calculate the correction factor
    #correction_factor = model_mean - obs_mean
    sigma= model_std+obs_std
    mu= model_mean - obs_mean * sigma

    # Apply the correction factor to the model data
    model_biascor = (model - mu)/sigma
    return model_biascor




def variance_correction(obs, model):
    """
    Perform variance bias correction on the model data.
    
    Parameters:
    obs (np.array): Observed values, shape (n_months,)
    model (np.array): Model forecast values, shape (n_ensembles, n_months)
    
    Returns:
    np.array: Bias-corrected model values, same shape as input model
    """
    # Calculate the mean across all ensembles and months
    model_std = np.std(model)
    obs_std = np.std(obs)
    # Calculate the correction factor
    #correction_factor = model_mean - obs_mean
    sigma= model_std/obs_std
    #mu= model_mean - obs_mean * sigma

    # Apply the correction factor to the model data
    model_biascor = model/sigma
    return model_biascor




def kurtosis_power_transform(data, power=0.5):     
    # model type
    original_dtype = data.dtype
    data = data.astype(np.float64)    
    # mean and standard deviation
    mu = np.mean(data, axis=1, keepdims=True)
    sigma = np.std(data, axis=1, keepdims=True)   
    # standardisation
    standardised = (data - mu) / sigma   
    # non-linear transformation
    sign = np.sign(standardised)
    abs_std = np.abs(standardised)
    transformed = sign * (abs_std ** power)
    transformed_std = np.std(transformed, axis=1, keepdims=True)
    transformed = transformed / transformed_std   
    # get original mean and standard deviation back
    transformed = transformed * sigma + mu
    transformed = transformed.astype(original_dtype)  
    return transformed



def qmap(obs, mod):
    obs_sorted = np.sort(obs)
    mod_sorted = np.sort(mod)
    obs_quantiles = np.linspace(0, 1, len(obs_sorted))
    mod_quantiles = np.linspace(0, 1, len(mod_sorted))
    interp_func = np.interp(mod_quantiles, obs_quantiles, obs_sorted)
    def map_func(x):
        indices = np.searchsorted(mod_sorted, x)
        return interp_func[np.minimum(indices, len(interp_func) - 1)]    
    return map_func


def Quantile_mapping(obs,mod):
    # quantile mapping method
    mod_data = mod.data
    obs_data = obs.data   
    def calculate_return_periods(data, n_years=30):
        sorted_data = np.sort(data)[::-1]
        ranks = np.arange(1, len(data) + 1)
        return_periods = (n_years * (len(data) / n_years) + 1) / ranks
        return sorted_data, return_periods

    obs_sorted, obs_return_periods = calculate_return_periods(obs_data)
    return_periods = np.logspace(0, np.log10(30*80), 100)
    # Qmap
    qmap_func = qmap(obs_data, mod_data.flatten())
    unseen_qmap = qmap_func(mod_data)


    unseen_qmap_flat = unseen_qmap.flatten()
    unseen_sorted, unseen_return_periods = calculate_return_periods(mod_data.flatten())
    unseen_qmap_sorted, unseen_qmap_return_periods = calculate_return_periods(unseen_qmap_flat)
    
    # create model cube
    time_coord = coords.DimCoord(range(unseen_qmap.shape[0]), standard_name='time')
    if unseen_qmap.ndim > 1:
        year_coord = coords.DimCoord(range(unseen_qmap.shape[1]), standard_name='realization')
        year_coord.units = cf_units.Unit('years since 1993', calendar='gregorian')
        mod_qmap = iris.cube.Cube(unseen_qmap, dim_coords_and_dims=[(time_coord, 0), (year_coord, 1)])
    else:
        mod = iris.cube.Cube(unseen_qmap, dim_coords_and_dims=[(time_coord, 0)])
    mod_qmap.long_name = 'Hunan_Jan_tas_model_DePreSys3 qmap'
    mod_qmap.units = 'celsius'
    mod_qmap
    #plot 
    plt.figure(figsize=(8, 6))
    plt.scatter(obs_return_periods, obs_sorted, label='Observed', color='blue', alpha=0.7, s=50)
    plt.scatter(unseen_return_periods, unseen_sorted, label='UNSEEN', color='red', alpha=0.3, s=20)
    plt.scatter(unseen_qmap_return_periods, unseen_qmap_sorted, label='UNSEEN Qmap', color='grey', alpha=0.3, s=20)
    plt.xscale('log')
    plt.xlabel('Return period (years)', fontsize=12)
    plt.ylabel('Average Values', fontsize=12)
    plt.title('Extreme_value_plot_corrections_obs_unseen', fontsize=12)
    x_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,2400]
    plt.xticks(x_ticks, x_ticks, fontsize=10)
    plt.legend(fontsize=10)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    return mod_qmap
    
    
def ks_compare_distributions(obs, mod, mod_cor, n_series=10000):
    def generate_random_series(mod, n_series):
        m, n = mod.shape
        random_series = np.zeros((n_series, m))
        for i in range(n_series):
            random_columns = np.random.randint(0, n, m)
            random_series[i] = mod[np.arange(m), random_columns]
        return random_series

    def perform_ks_tests(obs, random_series):
        ks_statistics = []
        p_values = []
        for series in random_series:
            ks_statistic, p_value = stats.ks_2samp(obs, series)
            ks_statistics.append(ks_statistic)
            p_values.append(p_value)
        return np.array(ks_statistics), np.array(p_values)

    def plot_results(statistics_mod, p_values_mod, statistics_mod_cor, p_values_mod_cor):
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle("KS test result: comparison of mod and mod_cor with obs", fontsize=25)
        plt.rc('axes', titlesize=20)
        plt.rc('axes', labelsize=20)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.rc('legend', fontsize=18)

        # K-S Statistics Distribution
        sns.histplot(statistics_mod, kde=True, ax=axes[0, 0], color='blue', alpha=0.5, label='mod')
        sns.histplot(statistics_mod_cor, kde=True, ax=axes[0, 0], color='red', alpha=0.5, label='mod_cor')
        axes[0, 0].set_title('Distribution of K-S Statistics')
        axes[0, 0].set_xlabel('K-S Statistic')
        axes[0, 0].legend()

        # p-values Distribution
        sns.histplot(p_values_mod, kde=True, ax=axes[0, 1], color='blue', alpha=0.5, label='mod')
        sns.histplot(p_values_mod_cor, kde=True, ax=axes[0, 1], color='red', alpha=0.5, label='mod_cor')
        axes[0, 1].set_title('Distribution of p-values')
        axes[0, 1].set_xlabel('p-value')
        axes[0, 1].legend()

        # K-S Statistic vs p-value
        axes[1, 0].scatter(statistics_mod, p_values_mod, alpha=0.1, color='blue', label='mod')
        axes[1, 0].scatter(statistics_mod_cor, p_values_mod_cor, alpha=0.1, color='red', label='mod_cor')
        axes[1, 0].set_title('K-S Statistic vs p-value')
        axes[1, 0].set_xlabel('K-S Statistic')
        axes[1, 0].set_ylabel('p-value')
        axes[1, 0].legend()

        # ECDF of p-values
        sns.ecdfplot(p_values_mod, ax=axes[1, 1], color='blue', label='mod')
        sns.ecdfplot(p_values_mod_cor, ax=axes[1, 1], color='red', label='mod_cor')
        axes[1, 1].set_title('ECDF of p-values')
        axes[1, 1].set_xlabel('p-value')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def print_summary(statistics, p_values, name):
        print(f"Summary for {name}:")
        print(f"Mean K-S Statistic: {np.mean(statistics):.4f}")
        print(f"Mean p-value: {np.mean(p_values):.4f}")
        print(f"Proportion of significant results (p < 0.05): {np.mean(p_values < 0.05):.2%}")

    # Process mod
    random_series_mod = generate_random_series(mod, n_series)
    statistics_mod, p_values_mod = perform_ks_tests(obs, random_series_mod)

    # Process mod_cor
    random_series_mod_cor = generate_random_series(mod_cor, n_series)
    statistics_mod_cor, p_values_mod_cor = perform_ks_tests(obs, random_series_mod_cor)

    # Plot results
    plot_results(statistics_mod, p_values_mod, statistics_mod_cor, p_values_mod_cor)

    # Print summaries
    print_summary(statistics_mod, p_values_mod, "mod")
    print("\n")
    print_summary(statistics_mod_cor, p_values_mod_cor, "mod_cor")
    
    
def distance_correlation(X, Y):
    def centering(D):
        mean_D = np.mean(D)
        n = D.shape[0]
        return D - np.mean(D, axis=0) - np.mean(D, axis=1)[:, None] + mean_D

    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    
    if X.shape[0] != Y.shape[0]:
        raise ValueError('X and Y must have the same length')
    
    n = X.shape[0]
    X = X.reshape(n, -1)
    Y = Y.reshape(n, -1)
    X_dist = squareform(pdist(X))
    Y_dist = squareform(pdist(Y))
    X_cent = centering(X_dist)
    Y_cent = centering(Y_dist)
    numerator = np.sum(X_cent * Y_cent)
    denominator = np.sqrt(np.sum(X_cent ** 2) * np.sum(Y_cent ** 2))
    
    if denominator > 0:
        return numerator / denominator
    else:
        return 0

def calculate_correlations(obs, mod):
    correlations = np.zeros((mod.shape[1], 2))
    for i in range(mod.shape[1]):
        correlations[i, 0] = distance_correlation(obs[:, 0], mod[:, i, 0])  # 降水
        correlations[i, 1] = distance_correlation(obs[:, 1], mod[:, i, 1])  # 温度
    return correlations

def mahalanobis_distance(x, data):
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    inv_cov = np.linalg.inv(cov)
    return mahalanobis(x, mean, inv_cov)



def calculate_return_periods(data, n_years=30):
    sorted_data = np.sort(data)[::-1]
    ranks = np.arange(1, len(data) + 1)
    return_periods = (n_years * (len(data) / n_years) + 1) / ranks
    return sorted_data, return_periods

def plot_return_periods(obs_data, models_data, model_names, n_years=30,m_ensemble = 100):
    obs_sorted, obs_return_periods = calculate_return_periods(obs_data)
    
    return_periods = np.logspace(0, np.log10(n_years * m_ensemble), 100)
    
    plt.figure(figsize=(12, 8))
    
    # Plot observed data
    plt.scatter(obs_return_periods, obs_sorted, label='Observed', color='blue', alpha=0.7, s=50)
    
    for mod_data, name in zip(models_data, model_names):
        mod_data_flat = mod_data.flatten()      
        # Calculate return periods for the model data
        mod_sorted, mod_return_periods = calculate_return_periods(mod_data_flat)        
        # Plotting model data
        plt.scatter(mod_return_periods, mod_sorted, label=f'{name}', alpha=0.3, s=20)
    
    plt.xscale('log')
    plt.xlabel('Return period (years)', fontsize=12)
    plt.ylabel('Average Precipitation(mm)', fontsize=12)
    plt.title('Extreme Value Plot with Corrections', fontsize=12)
    x_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000,3000]
    plt.xticks(x_ticks, x_ticks, fontsize=10)
    plt.legend(fontsize=10)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
