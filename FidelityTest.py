#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pdb

import numpy as np
import scipy.stats as stats
import random
import matplotlib.pyplot as plt
import math
import numpy.ma as ma
import datetime as dt
import iris.plot as iplt
import matplotlib as mpl
import iris


# In[4]:


def calcPercentile(obs, model): # returning the location percentage of obs/model
    '''
    Parameters
    ----------
    obs:        observed values;    list/array
    model:      modelled values;    list/array

    Returns
    -------
    percentile: where obs lie in model range
     '''
    model = sorted(model)
    ind = min(range(len(model)), key=lambda i: abs(model[i]-obs))
    percentile = ind/float(len(model))*100
    return percentile


# In[5]:


def calcDistrStatistic(obs, model, statistics, n_samples):
    '''
    Parameters
    ----------
    obs:        observed values;    numpy array, dimension (nyrs[, nmonths])
    model:      modelled values;    numpy array, dimension (nyrs, n ensemble members[, nmonths])
    n_samples:  number of samples;  integer
    statistics: statistics to calculate (mean, std, skew, kurt, linear_trend); tuple of strings

    Returns
    -------
    obs_stat:   Observed distribution mean
    model_stat: n_samples model distribution statistic

    Note: Sampling retains the time series
    '''
    nyrs = obs.shape[0]
    # test if there are multiple months being tested, i.e. lumped together
    # if so, determine number of months, flatten obs,
    # reshape the model array to be [nyrs, ensemble members x nmonths]
    if obs.ndim != 1:
        nmonths = obs.shape[1]
        obs = obs.ravel()
        model = model.reshape(model.shape[0], model.shape[1]*model.shape[2])
    else:
        nmonths = 1
        # Added next line for testing
        model = model.reshape(model.shape[0], model.shape[1])


    obs_stats_dict = {}

    # calculate obs distribution statistic
    if 'mean' in statistics:
        obs_stats_dict['mean'] = np.mean(obs)
    if 'std' in statistics:
        obs_stats_dict['std'] = np.std(obs)
    if 'skew' in statistics:
        obs_stats_dict['skew'] = stats.skew(obs)
    if 'kurt' in statistics:
        obs_stats_dict['kurt'] = stats.kurtosis(obs)
    if 'linear_trend' in statistics:
        obs_stats_dict['linear_trend'] = np.polyfit(np.arange(len(obs)), obs, 1)[0]
    else:
        raise UserWarning('stats test not defined')

    ## loop over number of samples, sampling from model ensemble to generate 'time series'
    ## same length as obs (i.e. nyrs*nmonths), and calculate n_samples of the distribution statistic

    model_stats_dict = {}

    for stat in statistics:
        model_stats_dict[stat] = []

    for _ in range(n_samples):
        model_sample = []
        for iyr in range(nyrs):
            sample_index = random.sample(range(model.shape[1]), nmonths)
            model_sample.append(model[iyr, sample_index])
        model_sample = np.concatenate(model_sample)
        if 'mean' in statistics:
            model_stats_dict['mean'].append(np.mean(model_sample))
        if 'std' in statistics:
            model_stats_dict['std'].append(np.std(model_sample))
        if 'skew' in statistics:
            model_stats_dict['skew'].append(stats.skew(model_sample))
        if 'kurt' in statistics:
            model_stats_dict['kurt'].append(stats.kurtosis(model_sample))
        if 'linear_trend' in statistics:
            model_stats_dict['linear_trend'].append(np.polyfit(np.arange(len(model_sample)), model_sample, 1)[0])

    return obs_stats_dict, model_stats_dict


# In[6]:


def timeseries_fid_test(obs, model):
    '''
    Parameters
    ----------
    obs:        observed values;    Iris cube dimension (nyrs[, nmonths])
    model:      modelled values;    Iris cube dimension (nyrs, n ensemble members[, nmonths])

    Returns
    -------
    obs_stats_dict: dictionary of observations distribution statistics
    mod_stats_dict: dictionary of n_samples model distribution statistics
    mean_perc: where obs lie in model range
    std_perc: where obs lie in model range
    skew_perc: where obs lie in model range
    kurt_perc: where obs lie in model range
    '''
    print('Fidelity testing')
    # calculate distribution statistics
    print('calculating distribution statistics')
    N_SAMPLES     = 10000   # number of proxy timeseries used
    obs_stats_dict, mod_stats_dict = calcDistrStatistic(obs, model, ['mean', 'std', 'skew', 'kurt', 'linear_trend'], N_SAMPLES)

    # Outputs where the obs lies as a percentage of the modelled values
    print('calculating percentiles')
    mean_perc = calcPercentile(obs_stats_dict['mean'], mod_stats_dict['mean'])
    std_perc  = calcPercentile(obs_stats_dict['std'], mod_stats_dict['std'])
    skew_perc = calcPercentile(obs_stats_dict['skew'], mod_stats_dict['skew'])
    kurt_perc = calcPercentile(obs_stats_dict['kurt'], mod_stats_dict['kurt'])
    linear_trend_perc = calcPercentile(obs_stats_dict['linear_trend'], mod_stats_dict['linear_trend'])

    print('Mean perc: ' + str(mean_perc))
    print('Standard deviation perc: ' + str(std_perc))
    print('Skewness perc: ' + str(skew_perc))
    print('Kurtosis perc: ' + str(kurt_perc))
    print('Linear trend perc: ' + str(linear_trend_perc))

    return [obs_stats_dict, mod_stats_dict, mean_perc, std_perc, skew_perc, kurt_perc]


# In[7]:


def plotStatsMeasures(obs_stat, mod_stat, stat_perc, posrow, poscol, title, xticks):
    NROWS = 2
    NCOLS = 4
    ax = plt.subplot2grid((NROWS, NCOLS), (posrow, poscol), colspan=1, rowspan=1)
    n = plt.hist(mod_stat, bins=21, histtype='stepfilled', 
                 color='r', alpha=0.5, label='Mod')
    plt.axvline(obs_stat, color='k', label='Obs')
    
    # 设置标题字体大小
    plt.title('{}, {:.0f}%'.format(title, stat_perc), fontsize=12)
    
    plt.yticks([], [])
    
    # 设置 x 轴刻度标签和字体大小
    plt.xticks(fontsize=10)
    
    # 设置坐标轴标签字体大小（如果需要）
    #plt.xlabel("X轴标签", fontsize=10)
    
    # 设置图例字体大小
    plt.legend(fontsize=10)
    
    # 调整子图以适应标题和标签
    plt.tight_layout()


# In[8]:


def get_cube_limits(data):
    '''A function to return the min and max values from a cube

    Input args:
    ----------
    cube - cube of data

    Returns:
    ----------
    vmin - the minimum value in the cube
    vmin - the maximum value in the cube
    '''
    vmin = data.min()
    vmax = data.max()
    return vmin, vmax


# In[9]:


def plot_fidelity_testing(obs, model, stats_measures, step, title, fname):
    plt.rcParams.update({'font.size': 12})
    mod_min, mod_max = get_cube_limits(model)
    obs_min, obs_max = get_cube_limits(obs)
    xmin = math.trunc(min(mod_min.astype('float'), obs_min.astype('float')))
    xmax = math.ceil(max(mod_max.astype('float'), obs_max.astype('float')))

    print('plotting')
    fig = plt.figure(figsize=(10., 5.))
    
    NROWS = 2
    NCOLS = 4
    ax = plt.subplot2grid((NROWS, NCOLS), (0, 0), colspan=2, rowspan=2)
    ax = plt.hist(model.ravel(), bins=10, density=True, stacked=True, histtype='stepfilled', \
                  color='salmon', label='Mod')
    ax = plt.hist(obs.ravel(), bins=10, density=True, stacked=True, histtype='stepfilled', \
                  color='k', alpha=.5, label='Obs')

    plt.yticks([])
    LEG_LOC = 'upper right'
    plt.legend(loc=LEG_LOC, prop={'size': 14})
    STAT_TITLES = ['Mean', 'StdDev', 'Skewness', 'Kurtosis']

    obs_mean, mod_mean, mean_perc = stats_measures[0]['mean'], stats_measures[1]['mean'], stats_measures[2]
    obs_std, mod_std, std_perc = stats_measures[0]['std'], stats_measures[1]['std'], stats_measures[3]
    obs_skew, mod_skew, skew_perc = stats_measures[0]['skew'], stats_measures[1]['skew'], stats_measures[4]
    obs_kurt, mod_kurt, kurt_perc = stats_measures[0]['kurt'], stats_measures[1]['kurt'], stats_measures[5]

    plotStatsMeasures(obs_mean, mod_mean, mean_perc, 0, 2, STAT_TITLES[0], np.arange(xmin, xmax, step))
    plotStatsMeasures(obs_std, mod_std, std_perc, 1, 2, STAT_TITLES[1], [0, 0.5, 1])
    plotStatsMeasures(obs_skew, mod_skew, skew_perc, 0, 3, STAT_TITLES[2], [-2, 0, 2])
    plotStatsMeasures(obs_kurt, mod_kurt, kurt_perc, 1, 3, STAT_TITLES[3], [-2, 0, 2, 4])

    # subplot spacing
    LEFT = 0.05
    BOTTOM = 0.05
    RIGHT = 0.97
    TOP = 0.85
    WSPACE = 0.35
    HSPACE = 0.28
    plt.subplots_adjust(left=LEFT, bottom=BOTTOM, right=RIGHT, top=TOP,
                        wspace=WSPACE, hspace=HSPACE)
    plt.suptitle(title)
    plt.show()


def fidelity_testing_compare(obs, mod, mod_cor, stats_measures, stats_measures_cor, step, title, fname):
    fig = plt.figure(figsize=(16, 12))

    # 主分布图
    ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
    ax_main.hist(mod.data.ravel(), bins=20, density=True, alpha=0.5, histtype='stepfilled', 
                 color='red', label='Mod')
    ax_main.hist(mod_cor.data.ravel(), bins=20, density=True, alpha=0.5, histtype='stepfilled', 
                 color='lightblue', label='Mod (Corrected)')
    ax_main.hist(obs.data.ravel(), bins=20, density=True, alpha=0.5, histtype='stepfilled', 
                 color='k', label='Obs')
    ax_main.set_yticks([])
    ax_main.legend(loc='upper right', prop={'size': 12})
    ax_main.set_title('Distribution Comparison', fontsize=14)
    ax_main.set_xlabel('Value', fontsize=12)
    STAT_TITLES = ['Mean', 'StdDev', 'Skewness', 'Kurtosis']
    for i, stat in enumerate(['mean', 'std', 'skew', 'kurt']):
        ax = plt.subplot2grid((3, 4), (2, i))
        
        obs_stat = stats_measures[0][stat]
        mod_stat = stats_measures[1][stat]
        mod_cor_stat = stats_measures_cor[1][stat]
        
        ax.hist(mod_stat, bins=20, density=True, alpha=0.5, histtype='stepfilled', 
                color='red', label='Mod')
        ax.hist(mod_cor_stat, bins=20, density=True, alpha=0.5, histtype='stepfilled', 
                color='lightblue', label='Mod (Corrected)')
        ax.axvline(obs_stat, color='k', linestyle='--', label='Obs')
        
        ax.set_title(STAT_TITLES[i])
        ax.set_yticks([])
        if i == 0:
            ax.legend(loc='upper right', prop={'size': 8})

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(title, fontsize=12)
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()



