# ------------------------------------------------------------------------------
import gc
import pandas as pd
import numpy as np
import scipy.linalg as la
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize
from sklearn.feature_selection import f_regression, mutual_info_regression
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import time
import sys
import datetime

import plotting
import pydotplus
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
from IPython.display import Image
import stats

# for model tree regression
from ModelTree import ModelTree
from linear_regr import linear_regr

import alphasense as alp
# ------------------------------------------------------------------------------
CONF_DECIMALS = 6
CONF_REG = 'mtr'
# ------------------------------------------------------------------------------
# list of all figures plotted
no2_figs = []
o3_figs = []
no2_fignames = []
o3_fignames = []
# ==============================================================================
# Statistics routines
def preclean_df(X, cols=None):

  sizex = len(X)

  print "Pre-cleaning infinite and negative values (set_size %d)" % sizex
  X = X.replace([np.inf, -np.inf], np.nan)
  if cols is not None:
    y = X.iloc[:, cols]

    # remove non-positive ppb values
    X = X[y.applymap(lambda x: x > 0 or np.isnan(x)).all(1)]
  else:
    X = X[X.applymap(lambda x: x > 0 or np.isnan(x)).all(1)]

  print "%d negative entries dropped" % (sizex - len(X))
  return X

# ------------------------------------------------------------------------------
def clean_data(X, sigma_mult, clean_ref=False):
  '''
     Clean data beyond sigma_mult standard deviations from the mean
     in each column of X

     Parameters:
      X          - Dataset to clean
      sigma_mult - Multiplying factor

     Return:
      X - cleaned dataset
  '''
  
  print "Cleaning x+-%dsigma" % sigma_mult
  sizex = len(X)

  # remove values beyond +-sigma_mult * sigma
  mu = np.mean(X.values, axis=0)
  sigma = np.std(X.values, axis=0)
  
  ranges = [(mu - sigma_mult * sigma).tolist(),
            (mu + sigma_mult * sigma).tolist()]

  if clean_ref:
    print "Cleaning reference monitor data as well!"
    for i in xrange(1, X.values.shape[1]):
      X = X[X.iloc[:,[i]].applymap(lambda x: (x > ranges[0][i]
          and x < ranges[1][i])).all(1)]
  else:
    for i in xrange(3, X.values.shape[1]):
      X = X[X.iloc[:,[i]].applymap(lambda x: (x > ranges[0][i]
          and x < ranges[1][i])).all(1)]

  print "%d entries dropped" % (sizex - len(X))
  print len(X)

  return X
# ------------------------------------------------------------------------------
def window_avg(data_df, window_size):
  '''
    Average contiguous chunks of data with size, window_size

    NOTE:  Features are assumed to be columns, 1st column has timestamp
           epochs
  '''
  vals = data_df.values
  avg_vals = []

  if window_size == 1:
    return data_df

  i = 0
  k = 0
  while i < vals.shape[0]:
    start_t = vals[i, 0]

    # record boundary in wd_bdry
    wd_bdry = i + 1
    for j in range(i + 1, vals.shape[0]):
      if (vals[j, 0] - start_t) >= (window_size * 60):
        wd_bdry = j
        break

    #print k, i, sec_bdry
    window = np.nanmean(vals[range(i, wd_bdry), 1:], axis=0)
    avg_vals.append(window.tolist())
    avg_vals[k].insert(0, start_t)
    i = wd_bdry
    k = k + 1

  avg_vals = pd.DataFrame(avg_vals).dropna()

  return avg_vals
# ==============================================================================
# Routines for visualization
def watermark(ax, loc, add_string):

  ts = time.time()
  st = datetime.datetime.fromtimestamp(ts).strftime('%Y/%m/%d %H:%M:%S')
  txt = r'\textit{' + add_string + '}\n'
  txt += r'\textbf{Deployed Location}: ' + loc + '\n'
  txt += r'\textbf{Plotted on}: ' + st

  return txt
# ------------------------------------------------------------------------------
def get_corr_txt(y_true, y_pred, add_title=''):

  if np.size(y_true) == np.shape(y_true)[0]:
    y_true = np.reshape(y_true, [np.size(y_true), 1])

  if np.size(y_pred) == np.shape(y_pred)[0]:
    y_pred = np.reshape(y_pred, [np.size(y_pred), 1])

  rmse = stats.rmse(y_true, y_pred)
  mape = stats.mape(y_true, y_pred)
  mae  = stats.mae(y_true, y_pred)
  pearson = stats.pearson(y_true, y_pred)
  coeff_det = stats.coeff_deter(y_true, y_pred)

  text = r'\textbf{Correlation Stats %s}'% add_title
  text = text + '\n' + r'$ R^2  = %g $' % coeff_det
  text = text + '\n' + r'$ MAE  = %g $' % mae
  text = text + '\n' + r'$ RMSE = %g $' % rmse
  text = text + '\n' + r'$ MAPE = %g\%% $' % mape
  text = text + '\n' + r'$ r_P  = %g $' % pearson

  return text
# ------------------------------------------------------------------------------
def visualize_rawdata(epochs, no2_x, no2_y, ox_x, o3_y):

  # visualize time-series of ref data
  fig, ax = plotting.ts_plot(epochs, no2_y,
         title=r'$ NO_2 $\textbf{ readings from Reference monitor}',
         ylabel=r'$ NO_2 $\textit{ concentration (ppb)}', ylim=[0, 100],
         leg_labels=[r'$ NO_2 $ conc (ppb)'], ids=[0])

  #fig = plotting.inset_hist_fig(fig, ax, no2_y, ['25%', '25%'], 1, ids=[0])
  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')

  no2_figs.append(fig)
  no2_fignames.append('no2-ref')

  fig, ax = plotting.ts_plot(epochs, o3_y,
         title=r'$ O_3 $ \textbf{readings from Reference monitor}',
         ylabel=r'$ O_3 $\textit{concentration (ppb)}', ylim=[0, 100],
         leg_labels=['$ O_3 $ conc (ppb)'], ids=[0])

  #fig = plotting.inset_hist_fig(fig, ax, o3_y, ['25%', '25%'], 1, ids=[0])
  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')

  o3_figs.append(fig)
  o3_fignames.append('o3-ref')

  # visualize time-series of AlphaSense sensors
  # TODO: Change labels based on sensor name 
  no2_op1_vals = np.zeros([np.shape(no2_x[0])[0], len(no2_x)])
  no2_op2_vals = np.zeros([np.shape(no2_x[0])[0], len(no2_x)])
  ox_op1_vals = np.zeros([np.shape(ox_x[0])[0], len(ox_x)])
  ox_op2_vals = np.zeros([np.shape(ox_x[0])[0], len(ox_x)])

  for (i, sens_no2) in enumerate(no2_x):
    no2_op1_vals[:, i] = sens_no2[:, 0]
    no2_op2_vals[:, i] = sens_no2[:, 1]

  ids=range(1, len(no2_x)+1)
  fig, ax = plotting.ts_plot(epochs, no2_op1_vals,
         title=r'$ NO_2 $ \textbf{Output1 from AlphaSense sensors}',
         ylabel=r'$ NO_2 $ \textit{op1 (mV)}', ylim=[210, 265],
         leg_labels=[('Sensor %d' % x) for x in range(1, len(no2_x)+1)],
         ids=ids)

  #fig = plotting.inset_hist_fig(fig, ax, no2_op1_vals,
  #                              ['25%', '25%'], 1, ids=ids)
  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')

  no2_figs.append(fig)
  no2_fignames.append('no2-op1')

  fig, ax = plotting.ts_plot(epochs, no2_op2_vals,
         title=r'$ NO_2 $ \textbf{Output2 from AlphaSense sensors}',
         ylabel=r'$ NO_2 $ \textit{op2 (mV)}', ylim=[210, 250],
         leg_labels=[('Sensor %d' % x) for x in range(1, len(no2_x)+1)],
         ids=ids)

  #fig = plotting.inset_hist_fig(fig, ax, no2_op2_vals,
  #                               ['25%', '25%'], 1, ids=ids)
  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')

  no2_figs.append(fig)
  no2_fignames.append('no2-op2')

  for (i, sens_ox) in enumerate(ox_x):
    ox_op1_vals[:, i] = sens_ox[:, 0]
    ox_op2_vals[:, i] = sens_ox[:, 1]

  fig, ax = plotting.ts_plot(epochs, ox_op1_vals,
         title=r'$ OX (NO_2 + O_3) $ \textbf{Output 1 from AlphaSense sensors}',
         ylabel=r'$ OX $ \textit{op1 (mV)}', ylim=[210, 265],
         leg_labels=[('Sensor %d' % x) for x in range(1, len(ox_x)+1)],
         ids=ids)

  #fig = plotting.inset_hist_fig(fig, ax, ox_op1_vals,
  #                              ['25%', '25%'], 1, ids=ids)
  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')

  o3_figs.append(fig)
  o3_fignames.append('o3-op1')

  fig, ax = plotting.ts_plot(epochs, ox_op2_vals,
         title=r'$ OX (NO_2 + O_3) $ \textbf{Output 2 from AlphaSense sensors}',
         ylabel=r'$ OX $ \textit{op2 (mV)}', ylim=[210, 250],
         leg_labels=[('Sensor %d' % x) for x in range(1, len(ox_x)+1)],
         ids=ids)

  #fig = plotting.inset_hist_fig(fig, ax, ox_op2_vals,
  #                              ['25%', '25%'], 1, ids=ids)
  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')

  o3_figs.append(fig)
  o3_fignames.append('o3-op2')
# ------------------------------------------------------------------------------
def plot_ts_prediction(epochs, no2_y, predict_no2, o3_y,
                       predict_o3, j, comp_witht=False, temps=None):

  # plot predicted vs. true ppb
  print "plotting actual and predicted values: NO2"
  t_series = np.array([no2_y, predict_no2]).T
  fig, ax = plotting.ts_plot(epochs, t_series,
        title = r'\textbf{True and Predicted concentrations of }'
              + r'$ NO_2 $ \textbf{ (Sensor %d)}' % (j + 1),
        ylabel = r'Concentration (ppb)',
        leg_labels=['Reference conc.', 'Predicted conc.'],
        ids=[0,(j+1)])

  text = get_corr_txt(t_series[:, 0], t_series[:, 1])
  ax.annotate(text, xy = (0.7, 0.75), xycoords='axes fraction')
  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')

  no2_figs.append(fig)
  no2_fignames.append('no2-sens%d-predict-true-comp' % (j+1))

  print "plotting actual and predicted values: O3"
  t_series = np.array([o3_y, predict_o3]).T
  fig, ax = plotting.ts_plot(epochs, t_series,
        title = r'\textbf{True and Predicted concentrations of } $ O_3 $',
        ylabel = r'Concentration (ppb)',
        leg_labels=['Reference conc.', 'Predicted conc.'],
        ids=[0, (j+1)])

  text = get_corr_txt(t_series[:, 0], t_series[:, 1])
  ax.annotate(text, xy = (0.7, 0.75), xycoords='axes fraction')
  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')

  o3_figs.append(fig)
  o3_fignames.append('o3-sens%d-predict-true-comp' % (j+1))

  # plot residuals wrt time
  print "plotting residual characteristics"
  ylim_p = [-150, 50]
  ylim_s = [0, 45]
  resid_no2 = no2_y - predict_no2
  resid_o3 = o3_y - predict_o3
  fig_n = None
  fig_o = None

  if not comp_witht:
    fig_n, ax = plotting.ts_plot(epochs, resid_no2,
          title=r"\textbf{Prediction errors (} $ NO_2 $ \textbf{) vs temperature (Sensor "
              + str(j + 1) + ")}",
          ylabel=r"\textit{Residuals (ppb)}", ylim=ylim_p,
          leg_labels=["Residual error"], ids=[(j+1)])
    #txt = watermark(ax, loc_label, '')
    #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')

    fig_o, ax = plotting.ts_plot(epochs, resid_o3,
          title=r"\textbf{Prediction errors (} $ O_X $ \textbf{) vs temperature (Sensor "
              + str(j + 1) + ")}",
          ylabel=r"\textit{Residuals (ppb)}", ylim=ylim_p,
          leg_labels=["Residual error"], ids=[(j+1)])
    #txt = watermark(ax, loc_label, '')
    #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')

  else:
    fig_n, ax = plotting.compare_ts_plot(epochs, resid_no2, temps,
          title=r"\textbf{Prediction errors (} $ NO_2 $ \textbf{) vs temperature (Sensor "
             + str(j + 1) + ")}",
          ylabel=r"\textit{Residuals (ppb)}",
          ylabel_s=r"\textit{Temperature} ($ ^{\circ} C $)", ylim_p=ylim_p,
          ylim_s=ylim_s, leg_labels=["Residual error", "Temperature"],
          ids=[(j+1), -1])

    # compute r^2 between residual and temperature
    p = np.polyfit(temps.astype(float), resid_no2.astype(float), 1)
    r = stats.pearson(temps.astype(float), resid_no2.astype(float))

    plot_str = "$ e = %g * T + %g $" % (stats.round_sig(p[0], 4),
          stats.round_sig(p[1], 4))
    plot_str2 = "$ {r}_{e,T} = %.4f $" % stats.round_sig(r, 4)

    ax.text((epochs[-1] + 3*epochs[0])/4, (ylim_p[0] + 19*ylim_p[1])/20,
            plot_str, ha="center", va="bottom")
    ax.text((epochs[-1] + 3*epochs[0])/4, (ylim_p[0] + 19*ylim_p[1])/20,
            plot_str2, ha="center", va="top")

    #ax.annotate("$ e = y_{pred} - y_{true} $", xy=(0.7, 0.9),
    #            xycoords="axes fraction")
    #txt = watermark(ax, loc_label, '')
    #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')
  
    # for O3
    fig_o, ax = plotting.compare_ts_plot(epochs, resid_o3, temps,
          title=r"\textbf{Prediction errors (} $ O_X $ \textbf{) vs temperature (Sensor "
              + str(j + 1) + ")}",
          ylabel=r"\textit{Residuals (ppb)}",
          ylabel_s=r"\textit{Temperature} ($ ^{\circ} C $)", ylim_p=ylim_p,
          ylim_s=ylim_s, leg_labels=["Residual error", "Temperature"],
          ids=[(j+1), -1])


    p = np.polyfit(temps.astype(float), resid_o3.astype(float), 1)
    r = stats.pearson(temps.astype(float), resid_o3.astype(float))

    plot_str = "$ e = %.3f * T + %.3f $" % (p[0], p[1])
    plot_str2 = "$ {r}_{e,T} = %.4f $" % r
    ax.text((epochs[-1] + 3*epochs[0])/4, (ylim_p[0] + 19*ylim_p[1])/20,
            plot_str, ha="center", va="bottom")
    ax.text((epochs[-1] + 3*epochs[0])/4, (ylim_p[0] + 19*ylim_p[1])/20,
            plot_str2, ha="center", va="top")
    
    ax.annotate("$ e = y_{pred} - y_{true} $", xy=(0.9, 0.9),
                xycoords="axes fraction")
    #txt = watermark(ax, loc_label, '')
    #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')
  
  no2_figs.append(fig_n)
  no2_fignames.append('no2-sens%d-res-temp-comp' % (j+1))
  
  o3_figs.append(fig_o)
  o3_fignames.append('o3-sens%d-res-temp-comp' % (j+1))

  # plot autocorrelation of residuals
  print "plotting autocorrelation of residuals"
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plotting.set_plot_labels(ax, title="Autocorrelation of $ NO_2 $ residuals",
      xlabel="Lag", ylabel=r"\textit{Autocorrelation}")
  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')
  fig = plot_acf(pd.Series(resid_no2).values, ax=ax, lags=np.arange(0, 2000, 10))

  no2_figs.append(fig)
  no2_fignames.append("no2-sens%d-autocorr" % (j+1))

  print "plotting autocorrelation of residuals"
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plotting.set_plot_labels(ax, title="Autocorrelation of $ O_3 $ residuals",
      xlabel="Lag", ylabel=r"\textit{Autocorrelation}")
  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')
  fig = plot_acf(pd.Series(resid_o3).values, ax=ax, lags=np.arange(0, 2000, 10))

  o3_figs.append(fig)
  o3_fignames.append("o3-sens%d-autocorr" % (j+1))
# ------------------------------------------------------------------------------
def plot_error_violins(metric, name='', full_name='', sens_type=''):

  # generate tick labels
  tick_labs = [('Train S%d' %x) for x in range(1, metric.shape[0]+1)]
  tick_labs.append('Train All')

  for i in xrange(metric.shape[0]):
    fig, ax = plotting.plot_violin(metric[i],
        title=r'\textbf{%s - } $ %s $ \textbf{ Sensor %d}' % (
                            name, sens_type, (i+1)),
        xlabel=r'\textit{Tested on Sensor %d}' % (i+1),
        ylabel=r'\textit{%s (%s)}' % (full_name, name),
        x_tick_labels=tick_labs)

    if sens_type == "NO_2":
      figs = no2_figs
      fignames = no2_fignames
    else:
      figs = o3_figs
      fignames = o3_fignames

    figs.append(fig)
    fignames.append("%s-%s-sens%d" % (sens_type.lower(), name.lower(), (i + 1)))
# ------------------------------------------------------------------------------
def plot_coeff_violins(coeff, names=[], sens_type=''):

  if len(names) == 0:
    names = ['var%d'%x for x in range(1, coeff.shape[2] + 1)]

  tick_labels = map(str, range(1, coeff.shape[1]))
  tick_labels.append('All')

  for i in xrange(coeff.shape[2]):
    fig, ax = plotting.plot_violin(coeff[:, :, i],
        title=r"\textbf{Coefficient of %s for } $ %s $" % (names[i], sens_type),
        ylabel=r"\textit{Coefficient of %s}" % names[i],
        xlabel=r"\textit{Trained on Sensor}",
        x_tick_labels=tick_labels)

    #txt = watermark(ax, loc_label, '')
    #ax.annotate(txt, xy=(0.6, 0.3), xycoords='axes fraction')

    if sens_type == "NO_2":
      figs = no2_figs
      fignames = no2_fignames
    else:
      figs = o3_figs
      fignames = o3_fignames

    figs.append(fig)
    fignames.append("%s-coeff%s" % (sens_type.lower(), names[i].lower()))
# ==============================================================================
# Routines for regression
def get_metrics(true_val, pred):
  metrics = []
  metrics.append(stats.mae(true_val, pred))
  metrics.append(stats.rmse(true_val, pred))
  metrics.append(stats.mape(true_val, pred))
  metrics.append(stats.coeff_deter(true_val, pred))
  metrics.append(stats.pearson(true_val, pred))
  metrics.append(np.mean(pred))
  metrics.append(np.std(pred))

  return metrics
# ------------------------------------------------------------------------------
def regress_once(X, X_t, y, labels=None, o3_y=None, no2_pred=None,
                 train_size=0.7, intercept=True, string=''):
  '''
     Perform regression on the given data-set and return coefficients
     and error metrics
    
     Parameters:
      X      - Values of X in linear regression
      y      - Values of y in linear regression
      labels - For computation of error only values corresponding to
               non-zero labels will be considered
      no2_pred - Prediction values for NO2 sensor (size same as y)
                  default - Don't subtract NO2 from output prediction
      train_size - ratio of the complete data-set used for training
      intercept - Include intercept in regression

     Return:
      coeffs - Coefficients of regression
      metrics - Array of metrics [MAE, RMSE, MAPE]
   '''
      
  
  train_size = int(np.floor(train_size * X.shape[0]))

  perm = np.arange(X.shape[0])
  perm = np.random.permutation(perm)
  if X_t is None:
    X_t = X

  X_ = X[perm, :]
  X_t = X_t[perm, :]
  y_ = y[perm]

  X_train = X_[:train_size, :]
  y_train = y_[:train_size]
  X_test = X_t[train_size:, :]

  if o3_y is not None:
    y_test = o3_y[perm][train_size:]
    no2_pred = no2_pred[perm][train_size:]
  else:
    y_test = y_[train_size:]

  # train model
  reg = LinearRegression(fit_intercept=intercept).fit(X_train, y_train)

  # test model
  if labels is not None:
    labels = labels[perm][train_size:]
    X_test = X_test[(labels != 0)]
    y_test = y_test[(labels != 0)]
    if no2_pred is not None:
      no2_pred = no2_pred[(labels != 0)]

  pred = reg.predict(X_test)
  if no2_pred is not None:
    pred = pred - no2_pred
  
  # plot of mean errors vs reference conc.
  #fig = plotting.plot_err_dist(y_test, pred,
  #            title=r'\textbf{Percentage errors of concentration ranges}',
  #            xlabel=r'\textit{Reference concentration (p.p.b.)}')

  #if o3_y is None:
  #  fig.savefig('/home/srijith/satvam/calib-data/inf/mriu-mape-dist-tst/no2-mape-ranges-'+string+'.pdf',
  #            format='pdf')
  #else:
  #  fig.savefig('/home/srijith/satvam/calib-data/inf/mriu-mape-dist-tst/o3-mape-ranges-'+string+'.pdf',
  #            format='pdf')

  metrics = get_metrics(y_test, pred)
  
  # training model
  #pred = reg.predict(X_train)
  #dev = stats.mae(y_train, pred)

  coeffs = reg.coef_.tolist()
  coeffs.append(reg.intercept_)

  return coeffs, metrics
# ------------------------------------------------------------------------------
def dtr_regress(X, X_t, y, labels=None, train_size=0.7,
                viz_tree=False, fname=None, optimize_depth=False,
                consider_mape=False):
  '''
     Perform regression on the given data-set and return coefficients
     and error metrics
    
     Parameters:
      X      - Values of X in linear regression
      X_t    - X for testing, if None uses random 30% from X
      y      - Values of y in linear regression
      labels - For computation of error only values corresponding to
               non-zero labels will be considered
      train_size - ratio of the complete data-set used for training
      viz_tree - Save an image of the regression tree if true
      fname  - Name used to save decision tree figures.  Only used if
               viz_tree is true
      optimize_depth - Use depth as hyperparameter and choose the optimum
                       depth for plots and visualization
      consider_mape  - Use MAPE as a metric for tuning hyper-parameters

     Return:
      metrics - Array of metrics [MAE, RMSE, MAPE, R^2, r, mu, sigma]
   '''

  train_size = int(np.floor(train_size * X.shape[0]))

  perm = np.arange(X.shape[0])
  perm = np.random.permutation(perm)
  if X_t is None:
    X_t = X

  X_ = X[perm, :]
  X_t = X_t[perm, :train_size]
  y_ = y[perm]

  X_train = X_[:train_size, :]
  y_train = y_[:train_size]
  X_test = X_t[train_size:, :]

  y_test = y_[train_size:]

  # test model
  if labels is not None:
    labels = labels[perm]
    labels = labels[train_size:]
    X_test = X_test[(labels != 0)]
    y_test = y_test[(labels != 0)]

  metrics1 = []
  metrics2 = []
  if optimize_depth:

    max_depths = np.arange(2, 20)
    min_err = np.zeros([2,])
    min_err[0] = np.inf
    min_err[1] = np.inf
    errs = np.zeros([np.size(max_depths), 2])

    for (i, depth) in enumerate(max_depths):
      # train model
      reg = DecisionTreeRegressor(criterion='mse',
              max_depth=depth)
      reg.fit(X_train, y_train)
      pred = reg.predict(X_test)

      errs[i, 0] = stats.rmse(y_test, pred)
      errs[i, 1] = stats.mape(y_test, pred)
      
      idx = 1 if consider_mape else 0
      idx2 = 1-idx
      if errs[i, idx] < min_err[idx]:
        metrics1 = get_metrics(y_test, pred)
        min_err[idx] = errs[i, idx]
        best_reg1 = reg
      if errs[i, idx2] < min_err[idx2]:
        metrics2 = get_metrics(y_test, pred)
        min_err[idx2] = errs[i, idx2]
        best_reg2 = reg

    fig, ax = plotting.plot_xy(max_depths, errs,
            title=r'\textbf{Error vs. Decision tree depth',
            xlabel=r'\textit{Tree depth}',
            ylabel='', leg_labels=['RMSE', 'MAPE'])

    fig.savefig('/media/srijith/Srijith/mriu-dtree/err-vs-depth/'+fname+'.svg', format='svg')
    #no2_figs.append(fig)
    #no2_fignames.append('rms-err-depth-dtr')
    #plt.show()

  else:
    best_reg = DecisionTreeRegressor(criterion='mse',
              max_depth=None)
    best_reg.fit(X_train, y_train)
    pred = best_reg.predict(X_test)
    metrics = get_metrics(y_test, pred)

  if viz_tree:
    dot_data = StringIO()
    if fname is None:
      fname = "dtree.png"

    export_graphviz(best_reg1, out_file=dot_data, filled=True,
                    rounded=True, special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(fname+'r.png')
  
    dot_data = StringIO()
    export_graphviz(best_reg2, out_file=dot_data, filled=True,
                    rounded=True, special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(fname+'m.png')
  
  #fig = plotting.plot_err_dist(y_test, pred,
  #            title=r'\textbf{Percentage errors of concentration ranges}',
  #            xlabel=r'\textit{Reference concentration (p.p.b.)}')

  #if o3_y is None:
  #  fig.savefig('/home/srijith/satvam/calib-data/inf/tmp/no2-mape-ranges-'+string+'.pdf',
  #            format='pdf')
  #else:
  #  fig.savefig('/home/srijith/satvam/calib-data/inf/tmp/o3-mape-ranges-'+string+'.pdf',
  #            format='pdf')

  return metrics1, metrics2
# ------------------------------------------------------------------------------
def model_tree_regress(X, X_t, y, sensor_type='no2', labels=None,
                       train_size=0.7, max_depth=4, min_samples_leaf=50):
  '''
     Use Model Tree Regression by Anson Wong to fit a linear
     regression model over nodes of a decision tree
  '''

  train_size = int(np.floor(train_size * X.shape[0]))

  perm = np.arange(X.shape[0])
  perm = np.random.permutation(perm)
  if X_t is None:
    X_t = X

  X_ = X[perm, :]
  X_t = X_t[perm, :]
  y_ = y[perm]

  X_train = X_[:train_size, :]
  y_train = y_[:train_size]
  X_test = X_t[train_size:, :]

  y_test = y_[train_size:]

  # test model
  if labels is not None:
    labels = labels[perm]
    labels = labels[train_size:]
    X_test = X_test[(labels != 0)]
    X_test2 = X_test[(labels == 0)]
    y_test = y_test[(labels != 0)]

  # try different parameters!
  regr = linear_regr()
  model_tree_regr = ModelTree(regr, max_depth=max_depth,
                              search_type='adaptive',
                              min_samples_leaf=min_samples_leaf)

  model_tree_regr.fit(X_train, y_train)

  pred = model_tree_regr.predict(X_test)
  pred2 = model_tree_regr.predict(X_test2)
  #nodes = []
  #ranges = []
  #vecs = []
  #for (i, tup) in enumerate(pred):
  #  node = tup[1]
  #  if node in nodes:
  #    j = nodes.index(node)
  #    vecs[j].append(y[i])
  #    if ranges[j][0] > y[i]:
  #      ranges[j][0] = y[i]
  #    if ranges[j][1] < y[i]:
  #      ranges[j][1] = y[i]
  #  else:
  #    nodes.append(node)
  #    vecs.append([y[i]])
  #    ranges.append([y[i], y[i]])

  #print nodes
  #print np.array(ranges)

  #print vecs
  #fig1 = plt.figure()
  #ax1 = fig1.add_subplot('111')

  #ax1.set_title(r'\textbf{Ranges of Reference concentration for leaf nodes of Model Tree}')
  #ax1.set_xlabel(r'\textit{Leaf nodes of Model Tree}')
  #ax1.set_ylabel(r'\textit{Reference conc. (ppb)')

  #ax1.yaxis.grid(b=True, which='major', linestyle='-', alpha=0.8)
  #ax1.xaxis.grid(b=True, which='major', linestyle='-', alpha=0.8)
  #ax1.yaxis.grid(b=True, which='minor', linestyle=':', alpha=0.5)
  #ax1.xaxis.grid(b=True, which='minor', linestyle=':', alpha=0.5)

  #medians = [np.median(vec) for vec in vecs]
  #ax1.boxplot(vecs, positions=medians)
  #ax1.set_xticklabels(['%0.1f'%x for x in medians], rotation='80')
  #fig1.savefig('/media/srijith/Srijith/mriu-dtree/modeltree-depth2/true-ranges-sens1-%s.pdf'%sensor_type, format='pdf')

  # plot of mean errors vs reference conc.
  #fig2 = plotting.plot_err_dist(y, [p[0] for p in pred],
  #            title=r'\textbf{Percentage errors of concentration ranges}',
  #            xlabel=r'\textit{Reference concentration (p.p.b.)}')

  #fig2.savefig('/media/srijith/Srijith/mriu-dtree/modeltree-depth3-samples150/mape-dist_sens1-%s.pdf'%sensor_type, format='pdf')

  #features = ['no2_op1', 'no2_op2', 'o3_op1', 'o3_op2'] \
  #                if (sensor_type == 'no2') else        \
  #           ['o3_op1', 'o3_op2', 'no2_op1', 'no2_op2']

  #model_tree_regr.export_graphviz('/media/srijith/Srijith/mriu-dtree/modeltree-depth3-samples150/modeltree_graph_%s_sens1' % sensor_type,
  #                                features, export_png=False, export_pdf=True)

  return get_metrics(y_test, pred), get_metrics(y_test2, pred2)
# ------------------------------------------------------------------------------
def tls_regress_once(X, y, labels=None, train_size=0.7, intercept=True):
  '''
     Perform total least squares regression on the given data-set and
     return coefficients and error metrics
    
     Parameters:
      X      - Values of X in linear regression
      y      - Values of y in linear regression
      labels - For computation of error only values corresponding to
               non-zero labels will be considered
      train_size - ratio of the complete data-set used for training
      intercept - Include intercept in regression

     Return:
      coeffs - Coefficients of regression
      metrics - Array of metrics [MAE, RMSE, MAPE]
   '''

  
  train_size = int(np.floor(train_size * X.shape[0]))

  perm = np.arange(X.shape[0])
  perm = np.random.permutation(perm)

  if intercept:
    X_ = np.vstack((X.T, np.ones(np.shape(X)[0]))).T

  X_ = X[perm, :]
  y_ = y[perm]

  X_train = X_[:train_size, :]
  y_train = y_[:train_size]
  X_test = X_[train_size:, :]
  y_test = y_[train_size:]
  n = np.shape(X_)[1]

  del X_, y_
  gc.collect()
  # train model
  # augmented matrix
  X_train = np.vstack((X_train.T, y_train)).T
  X_train, X_train, X_train = la.svd(X_train, full_matrices=True)

  Vxy = X_train.T[:n, n:]
  Vyy = X_train.T[n:, n:]

  coeffs = -Vxy / Vyy

  # test model
  if labels is not None:
    labels = labels[perm]
    labels = labels[train_size:]
    X_test = X_test[(labels != 0)]
    y_test = y_test[(labels != 0)]

  metrics = []
  pred = np.dot(X_test, coeffs)
  metrics.append(stats.mae(y_test, pred))
  metrics.append(stats.rmse(y_test, pred))
  metrics.append(stats.mape(y_test, pred))

  # training model
  # pred = np.dot(X_train, coeffs)
  # dev = stats.mae(y_train, pred)

  return coeffs, metrics
# ------------------------------------------------------------------------------
def rfr_regress_once(X, y, labels=None, train_size=0.7, intercept=True):
  
  train_size = int(np.floor(train_size * X.shape[0]))

  perm = np.arange(X.shape[0])
  perm = np.random.permutation(perm)

  X_ = X[perm, :]
  y_ = y[perm]

  X_train = X_[:train_size, :]
  y_train = y_[:train_size]
  X_test = X_[train_size:, :]
  y_test = y_[train_size:]

  # train model
  reg = RandomForestRegressor(n_estimators=10).fit(X_train, y_train)

  # test model
  if labels is not None:
    labels = labels[perm]
    labels = labels[train_size:]
    X_test = X_test[(labels != 0)]
    y_test = y_test[(labels != 0)]

  metrics = []
  pred = reg.predict(X_test)
  
  metrics.append(stats.mae(y_test, pred))
  metrics.append(stats.rmse(y_test, pred))
  metrics.append(stats.mape(y_test, pred))
  metrics.append(stats.coeff_deter(y_test, pred))
  metrics.append(stats.pearson(y_test, pred))

  # training model
  #pred = reg.predict(X_train)
  #dev = stats.mae(y_train, pred)
  #print reg.feature_importances_

  return metrics
# ------------------------------------------------------------------------------
def get_agg_sets(X, y, for_sens, n_pred=None, o3_y=None):
  '''
    Get a merged dataset containing all the matrices in X,
    and the corresponding output vector y, repeated.
    [assumption] - y is the same for all X's.
    
    Params:
      X     -   list of dataset matrices, size of each element
                is [num_examples, num_features]
      y     -   output vector of size [num_examples,]
      for_sens -  index of specific data to mark in labels
      n_pred  - [default None] prediction from NO2 sensors,
                length same as list length of X
      o3_y  - [default None] list of o3 reference values

    Returns:
      agg_X  - Aggregated data-set
      agg_y  - repeated output vector to match agg_X
      labels - Labels denoting locations of for_sens sensor
      agg_np - Aggregate no2_prediction
      agg_oy - aggregate o3_y vector after tiling appropriately
  '''

  agg_X = []
  agg_y = np.array([])

  if o3_y is not None:
    agg_oy = np.array([])
    agg_np = np.array([])
    assert len(o3_y) == len(n_pred)
  else:
    agg_oy = None
    agg_np = None

  labels = np.zeros([len(X) * np.shape(X[0])[0],])
  for (k, X_sensor) in enumerate(X):

    if k == 0:
      agg_X[:] = X_sensor 
    else:
      agg_X = np.concatenate((agg_X, X_sensor), axis=0)

    agg_y = np.concatenate((agg_y, y), axis=0)

    if o3_y is not None:
      agg_oy = np.concatenate((agg_oy, o3_y), axis=0)
      agg_np = np.concatenate((agg_np, n_pred), axis=0)

    if k == for_sens:
      labels[k * np.shape(X[0])[0] : (k+1) * np.shape(X[0])[0]] = 1

  assert len(labels) == len(agg_y)
  if agg_np is not None:
    assert len(agg_np) == len(agg_oy)
    assert len(agg_np) == len(labels)

  return agg_X, agg_y, labels, agg_np, agg_oy

# ------------------------------------------------------------------------------
def regress_sensor_type(X, y, sensor, epochs, train_ratio,
                        runs, reg='', no2_pred=None, o3_y=None):
  '''
     Train on all sensors of a certain type and obtain self
     trained and cross trained error measures

     Params:
      X       - list of training inputs for each sensor
      y       - training output values common to all
      sensor  - type of sensor. Any one of the following:
                'no2'   - NO2 sensor
                'ox'    - OX sensor
      train_ratio - Ratio of training set to consider
      runs    - Number of runs for each sensor
      reg     - algorithm to use for regression. Any one of the following:
                'tls'   - Total least squares
                'rfr'   - Random Forest regression
                default - Oridinary Least squares
      no2_pred - predicted NO2, None if sensor='no2'
      o3_y     - reference o3 values if sensor='ox'

     Return:
      coeffs, maes, rmses, mapes

      coeffs has shape [num_sensors, runs, (num_sensors + 1), num_coeffs]
      maes, rmses and mapes have shape [num_sensors, runs, (num_sensors + 1)]
  '''

  if sensor == 'ox' and (no2_pred is None or o3_y is None):
    raise ValueError("OX sensor regression: either NO2 prediction\
             or O3 output not specified")
  elif sensor == 'no2':
    no2_pred = None
    o3_y = None

  # coefficients for all steps carried out
  # shape: N * R * (N + 1) * m
  regress_sensor_type.coeffs = \
      np.zeros([len(X), runs, (len(X) + 1), (1+np.shape(X[0])[1])])

  # error metrics
  # shape: N * R * (N + 1)
  regress_sensor_type.maes = np.zeros([len(X), runs, (len(X) + 1)])
  regress_sensor_type.rmses = np.zeros([len(X), runs, (len(X) + 1)]) 
  regress_sensor_type.mapes = np.zeros([len(X), runs, (len(X) + 1)]) 
  regress_sensor_type.r2 = np.zeros([len(X), runs, (len(X) + 1)]) 
  regress_sensor_type.r = np.zeros([len(X), runs, (len(X) + 1)]) 
  regress_sensor_type.mean = np.zeros([len(X), runs, (len(X) + 1)]) 
  regress_sensor_type.std = np.zeros([len(X), runs, (len(X) + 1)]) 

  def _reg_metrics(X, X_test, y, pos, n_pred=None,
                   oy=None, lab=None, string=''):
    '''
      Regress and fill metrics and coeffs (if applicable)
    '''

    if reg == 'tls':
      tmpcoeffs, metrics = tls_regress_once(X, X_test, y,
                         train_size=train_ratio)

    elif reg == 'rfr':
      metrics = rfr_regress_once(X, X_test, y,
                         train_size=train_ratio)
      tmpcoeffs = None

    elif reg == 'mtr':
      metrics = model_tree_regress(X, X_test, y,
                        train_size=train_ratio, labels=lab)
      tmpcoeffs = None

    else:
      if sensor == 'ox':
        tmpcoeffs, metrics = regress_once(X, X_test, y,
                              train_size=train_ratio,
                              no2_pred=n_pred, o3_y=oy,
                              labels=lab, string=string)
      else:
        tmpcoeffs, metrics = regress_once(X, X_test, y,
                              train_size=train_ratio,
                              labels=lab, string=string)

    if tmpcoeffs is not None:
      regress_sensor_type.coeffs[pos[0], pos[1], pos[2], :] = tmpcoeffs
    else:
      regress_sensor_type.coeffs = None

    regress_sensor_type.maes[pos[0], pos[1], pos[2]] = metrics[0]
    regress_sensor_type.rmses[pos[0], pos[1], pos[2]] = metrics[1]
    regress_sensor_type.mapes[pos[0], pos[1], pos[2]] = metrics[2]
    regress_sensor_type.r2[pos[0], pos[1], pos[2]] = metrics[3]
    regress_sensor_type.r[pos[0], pos[1], pos[2]] = metrics[4]
    regress_sensor_type.mean[pos[0], pos[1], pos[2]] = metrics[5]
    regress_sensor_type.std[pos[0], pos[1], pos[2]] = metrics[6]

    return

  for j in xrange(len(X)):
    # build aggregate dataset
    if no2_pred is not None:
      agg_X, agg_y, labels, agg_np, agg_oy = \
                get_agg_sets(X, y, j, no2_pred[j][-1, :], o3_y)
    else:
      agg_X, agg_y, labels, agg_np, agg_oy = \
                get_agg_sets(X, y, j, n_pred=None, o3_y=None)
      
    print "Sensor: " + str(j + 1)
    for i in xrange(runs):

      sys.stdout.write("\rRun ........ %d (%.2f%% complete)"
                            % ((i+1), (i+1) * 100/runs))
      for k in xrange(len(X)):

        # Train on sensor k
        string = 'tr%dte%d' % (k+1, j+1)
        if no2_pred is not None:
          _reg_metrics(X[k], X[j], y, (j, i, k),
                       n_pred=no2_pred[j][k, :], oy=o3_y,
                       string=string)
        else:
          _reg_metrics(X[k], X[j], y, (j, i, k), string=string)

      # train on aggregate dataset
      string = 'trALLte%d' % (j+1)
      _reg_metrics(agg_X, None, agg_y, (j, i, -1),
                   n_pred=agg_np, oy=agg_oy,
                   lab=labels, string=string)

    print "\nSensor %d DONE" % (j+1)

  return regress_sensor_type.coeffs, regress_sensor_type.maes, \
         regress_sensor_type.rmses, regress_sensor_type.mapes, \
         regress_sensor_type.r2, regress_sensor_type.r, \
         regress_sensor_type.mean, regress_sensor_type.std
# ------------------------------------------------------------------------------
def preproc(data, temps_present, hum_present, incl_op1, incl_op2,
            incl_temps, incl_op1t, incl_op2t, incl_hum, incl_op1h, incl_op2h,
            incl_op12, incl_cross_terms, clean, runs, loc_label,
            training_set_ratio = 0.7):


  # remove outliers
  if clean is not None:
    data = clean_data(data, clean)

  data = data.values

  # remove possible outliers
  #tmp_data = pd.DataFrame(data.iloc[:, 0])
  #data = data[tmp_data.applymap(lambda x: x < 200).all(1)]

  train_size = int(np.floor(training_set_ratio * data.shape[0]))
  print "Training set size: \t", train_size
  print "Test set size: \t", (np.size(data, 0) - train_size)

  no2_x = []
  ox_x = []
  temp = []
  hum = []

  no2_y_pred = []
  o3_y_pred = []

  coeffs_no2_names = []
  coeffs_ox_names = []
  if incl_op1:
    coeffs_no2_names.append('no2op1')
    coeffs_ox_names.append('oxop1')
  if incl_op2:
    coeffs_no2_names.append('no2op2')
    coeffs_ox_names.append('oxop2')

  # column locations for no2, ox and temperature data of the ith sensor
  col_skip = 3

  if temps_present and hum_present:
    col_temp = (lambda i: (col_skip + 6*i))
    col_hum = (lambda i: (col_skip + 6*i + 1))
    col_no2 = (lambda i: range((col_skip + 6*i + 2),(col_skip + 6*i + 4)))
    col_ox = (lambda i: range((col_skip + 6*i + 4),(col_skip + 6*i + 6)))
  elif hum_present:
    col_hum = (lambda i: (col_skip + 5*i))
    col_no2 = (lambda i: range((col_skip + 5*i + 1),(col_skip + 5*i + 3)))
    col_ox = (lambda i: range((col_skip + 5*i + 3),(col_skip + 5*i + 5)))
  elif temps_present:
    col_temp = (lambda i: (col_skip + 5*i))
    col_no2 = (lambda i: range((col_skip + 5*i + 1),(col_skip + 5*i + 3)))
    col_ox = (lambda i: range((col_skip + 5*i + 3),(col_skip + 5*i + 5)))
  else:
    col_no2 = (lambda i: range((col_skip + 4*i),(col_skip + 4*i + 2)))
    col_ox = (lambda i: range((col_skip + 4*i + 2),(col_skip + 4*i + 4)))

  epochs = data[:,0]

  # store x and y values
  no2_y = data[:,1]
  o3_y = data[:,2]

  # convert o3 to ox for regression
  ox_y = o3_y + no2_y

  if incl_cross_terms:
    coeffs_no2_names.append('oxop1')
    coeffs_no2_names.append('oxop2')
    coeffs_ox_names.append('no2op1')
    coeffs_ox_names.append('no2op2')

  if temps_present:
    if incl_temps:
      coeffs_no2_names.append('temp')
      coeffs_ox_names.append('temp')

  if hum_present:
    if incl_hum:
      coeffs_no2_names.append('hum')
      coeffs_ox_names.append('hum')

  if temps_present:
    if incl_op1t:
      coeffs_no2_names.append('no2op1T')
      coeffs_ox_names.append('oxop1T')

    if incl_op2t:
      coeffs_no2_names.append('no2op2T')
      coeffs_ox_names.append('oxop2T')

  if hum_present:
    if incl_op1h:
      coeffs_no2_names.append('no2op1h')
      coeffs_ox_names.append('oxop1h')

    if incl_op2h:
      coeffs_no2_names.append('no2op2h')
      coeffs_ox_names.append('oxop2h')
      
  if incl_op12:
    coeffs_no2_names.append('no2op1no2op2')
    coeffs_ox_names.append('oxop1oxop2')
      
  coeffs_no2_names.append('constant')
  coeffs_ox_names.append('constant')
    
  for i in xrange(np.size(data, 1)):
    if col_ox(i)[-1] >= np.size(data, 1):
      break

    tmp_idx_n = []
    tmp_idx_o = []
    
    if incl_op1:
      tmp_idx_n.append(col_no2(i)[0])
      tmp_idx_o.append(col_ox(i)[0])
    if incl_op2:
      tmp_idx_n.append(col_no2(i)[1])
      tmp_idx_o.append(col_ox(i)[1])

    if incl_cross_terms:
      tmp_idx_n = np.concatenate((tmp_idx_n, col_ox(i)), axis=0).tolist()
      tmp_idx_o = np.concatenate((tmp_idx_o, col_no2(i)), axis=0).tolist()

    if temps_present:
      temp.append(data[:,col_temp(i)])
      if incl_temps:
        tmp_idx_n.append(col_temp(i))
        tmp_idx_o.append(col_temp(i))

    if hum_present:
      hum.append(data[:,col_hum(i)])
      if incl_hum:
        tmp_idx_n.append(col_hum(i))
        tmp_idx_o.append(col_hum(i))

    no2_x.append(data[:,tmp_idx_n])
    ox_x.append(data[:,tmp_idx_o])
 
    if temps_present and incl_op1t:
      no2_op1t = np.reshape(data[:, col_no2(i)[0]]
        * data[:, col_temp(i)], [np.shape(data)[0], 1])
      ox_op1t = np.reshape(data[:, col_ox(i)[0]]
        * data[:, col_temp(i)], [np.shape(data)[0], 1])

      no2_x[i] = np.concatenate((no2_x[i], no2_op1t), axis=1)
      ox_x[i] = np.concatenate((ox_x[i], ox_op1t), axis=1)

    if temps_present and incl_op2t:
      no2_op2t = np.reshape(data[:, col_no2(i)[1]]
        * data[:, col_temp(i)], [np.shape(data)[0], 1])
      ox_op2t = np.reshape(data[:, col_ox(i)[1]]
        * data[:, col_temp(i)], [np.shape(data)[0], 1])

      no2_x[i] = np.concatenate((no2_x[i], no2_op2t), axis=1)
      ox_x[i] = np.concatenate((ox_x[i], ox_op2t), axis=1)

    if hum_present and incl_op1h:
      no2_op1h = np.reshape(data[:, col_no2(i)[0]]
        * data[:, col_hum(i)], [np.shape(data)[0], 1])
      ox_op1h = np.reshape(data[:, col_ox(i)[0]]
        * data[:, col_hum(i)], [np.shape(data)[0], 1])

      no2_x[i] = np.concatenate((no2_x[i], no2_op1h), axis=1)
      ox_x[i] = np.concatenate((ox_x[i], ox_op1h), axis=1)

    if hum_present and incl_op2h:
      no2_op2h = np.reshape(data[:, col_no2(i)[1]]
        * data[:, col_hum(i)], [np.shape(data)[0], 1])
      ox_op2h = np.reshape(data[:, col_ox(i)[1]]
        * data[:, col_hum(i)], [np.shape(data)[0], 1])

      no2_x[i] = np.concatenate((no2_x[i], no2_op2h), axis=1)
      ox_x[i] = np.concatenate((ox_x[i], ox_op2h), axis=1)

    if incl_op12:
      no2_op12 = np.reshape(data[:, col_no2(i)[0]]
        * data[:, col_no2(i)[1]], [np.shape(data)[0], 1])
      ox_op12 = np.reshape(data[:, col_ox(i)[0]]
        * data[:, col_ox(i)[1]], [np.shape(data)[0], 1])

      no2_x[i] = np.concatenate((no2_x[i], no2_op12), axis=1)
      ox_x[i] = np.concatenate((ox_x[i], ox_op12), axis=1)

  #visualize_rawdata(epochs, no2_x, no2_y, ox_x, o3_y)

  return epochs, no2_x, no2_y, ox_x, o3_y, ox_y,\
         coeffs_no2_names, coeffs_ox_names
# ------------------------------------------------------------------------------
def regress_df(data, temps_present=False, hum_present=False,
               incl_op1=True, incl_op2=False,
               incl_temps=False, incl_op1t=False, incl_op2t=False,
               incl_hum=False, incl_op1h=False, incl_op2h=False,
               incl_op12=False, incl_cross_terms=False, clean=None,
               avg=None, runs=1000, save_fscores=False,
               loc_label='---', ret_errs=False):

  '''
    Regress the given dataframe with values for sensor features and
    features like temperature, humidity, etc.  This function also manages
    plotting the raw data and information obtained from it. Feature selection
    is left to the calling program, and can be done by configuring the
    switches in the args
  '''

  training_set_ratio = 0.7
  epochs, no2_x, no2_y, ox_x, o3_y, ox_y, coeffs_no2_names,\
         coeffs_ox_names = preproc(
              data, temps_present, hum_present,
              incl_op1, incl_op2, incl_temps, incl_op1t,
              incl_op2t, incl_hum, incl_op1h, incl_op2h,
              incl_op12, incl_cross_terms, clean,
              runs, loc_label, training_set_ratio)

  # process and regress data: multifold
  if CONF_REG == 'dtr':
    no2_metrics = []
    o3_metrics = []

    for (i, x) in enumerate(no2_x):
      no2_metrics.append(dtr_regress(x, None, y,
                                     train_size=training_set_ratio,
                                     viz_tree=True,
                                     fname='n-tree%d.png'%1,
                                     optimize_depth=True,
                                     consider_mape=True))
    for (i, x) in enumerate(ox_x):
      o3_metrics.append(dtr_regress(x, None, y,
                                   train_size=training_set_ratio,
                                   viz_tree=True,
                                   fname='o-tree%d.png'%i,
                                   optimize_depth=True,
                                   consider_mape=True))

    print no2_metrics
    print o3_metrics

  #elif CONF_REG == 'mtr':
  #  no2_metrics = []
  #  o3_metrics = []

  #  #for (i, x) in enumerate(no2_x):
  #  #x, y, lab, d2, d3 = get_agg_sets(no2_x, no2_y, 1)
  #  no2_metrics.append(model_tree_regress(no2_x[0], None, no2_y,
  #                                     #labels=lab,
  #                                     train_size=training_set_ratio,
  #                                     sensor_type='no2'))

  #  #for (i, x) in enumerate(ox_x):
  #  #x, y, lab, d2, d3 = get_agg_sets(ox_x, o3_y, 1)
  #  o3_metrics.append(model_tree_regress(ox_x[0], None, o3_y,
  #                                     #labels=lab,
  #                                     train_size=training_set_ratio,
  #                                     sensor_type='o3'))

  #  print no2_metrics
  #  print o3_metrics

  else:
    # always train no2 before o3
    print "\nFor NO_2 sensors....."
    coeffs_no2, maes_no2, rmses_no2, mapes_no2, r2_no2, r_no2,\
    means_no2, stds_no2 = regress_sensor_type(
          no2_x, no2_y, 'no2', epochs, training_set_ratio,
          runs, CONF_REG)

    # store no2 coefficients
    #no2_pred = []
    #for x in no2_x:
    #  x = np.concatenate((x, np.ones([x.shape[0],1])), axis=1)
    #  mean_coeffs_no2 = np.median(coeffs_no2, axis=0)
    #  mean_coeffs_no2 = np.median(mean_coeffs_no2, axis=0)
    #  #print mean_coeffs_no2
    #  no2_pred.append(np.dot(mean_coeffs_no2, x.T))    # shape: (num_sens + 1, data_len)

    print "\nFor OX sensors....."
    coeffs_ox, maes_ox, rmses_ox, mapes_ox, r2_ox, r_ox,\
    means_ox, stds_ox = regress_sensor_type(
          ox_x, o3_y, 'no2', epochs, training_set_ratio,
          runs, CONF_REG)#, no2_pred=no2_pred, o3_y=o3_y)

    # mean coefficients should have the shape (N+1) * m
    #mean_no2_coeffs = np.median(coeffs_no2, axis=(0, 1))
    #mean_ox_coeffs = np.median(coeffs_ox, axis=(0, 1))

    # plot violins for error metrics
    #plot_error_violins(maes_no2, name='MAE',
    #      full_name='Mean Absolute Error', sens_type='NO_2');
    #plot_error_violins(rmses_no2, name='RMSE',
    #      full_name='Root Mean Square Error', sens_type='NO_2');
    #plot_error_violins(mapes_no2, name='MAPE',
    #      full_name='Mean Absolute Percentage Error', sens_type='NO_2');
    #
    #plot_error_violins(maes_ox, name='MAE',
    #      full_name='Mean Absolute Error', sens_type='O_X');
    #plot_error_violins(rmses_ox, name='RMSE',
    #      full_name='Root Mean Square Error', sens_type='O_X');
    #plot_error_violins(mapes_ox, name='MAPE',
    #      full_name='Mean Absolute Percentage Error', sens_type='O_X');
    #
    # ravel the first two dimensions
    #coeffs_no2 = np.reshape(coeffs_no2,
    #      [coeffs_no2.shape[0] * coeffs_no2.shape[1],
    #       coeffs_no2.shape[2],
    #       coeffs_no2.shape[3]])
#
#    coeffs_ox = np.reshape(coeffs_ox,
#          [coeffs_ox.shape[0] * coeffs_ox.shape[1],
#           coeffs_ox.shape[2],
#           coeffs_ox.shape[3]])
#    
#    print coeffs_no2_names
#    print coeffs_ox_names
#
#    # plot violins for coefficients
#    plot_coeff_violins(coeffs_no2, names=coeffs_no2_names, sens_type='NO_2')
#    plot_coeff_violins(coeffs_ox, names=coeffs_ox_names, sens_type='O_3')

  # find p_vals of features
#  if save_fscores:
#    for (i, x) in enumerate(no2_x):
#      x = normalize(x, axis=0)
#      f, p = f_regression(x, no2_y)
#      f = np.concatenate((f, p), axis=1)
#      np.savetxt(DIR_PREFIX + ("fscores-no2-sens%d" % (i+1)),
#                 f, fmt='%0.4e', delimiter=',')
#
#    for (i, x) in enumerate(o3_x):
#      x = normalize(x, axis=0)
#      f, p = f_regression(x, ox_y)
#      f = np.concatenate((f, p), axis=1)
#      np.savetxt(DIR_PREFIX + ("fscores-ox-sens%d" % (i+1)),
#                 f, fmt='%0.4e', delimiter=',')

  # if needed in future....
  #mi = mutual_info_regression(x, no2_y)

  print np.median(maes_no2, axis=1)
  print np.median(rmses_no2, axis=1)
  print np.median(mapes_no2, axis=1)
  print np.median(r2_no2, axis=1)
  print np.median(r_no2, axis=1)

  print np.median(maes_ox, axis=1)
  print np.median(rmses_ox, axis=1)
  print np.median(mapes_ox, axis=1)
  print np.median(r2_ox, axis=1)
  print np.median(r_ox, axis=1)

  if ret_errs:
    return coeffs_no2, coeffs_ox, maes_no2, maes_ox, rmses_no2,\
           rmses_ox, mapes_no2, mapes_ox, r2_no2, r2_ox, r_no2, r_ox,\
           means_no2, means_ox, stds_no2, stds_ox
  else:
    return no2_figs, no2_fignames, o3_figs, o3_fignames

# ----------------------------------------------------------------------------------
def gen_global_err_set(dfs, grans=[1, 15, 60], runs=200, err_func=stats.mape):
  '''
    Generate a 7D numpy array of the shape
    [Loc x Gran x model x runs x senstype x test-set x metrics]

    dfs: list of dataframes
    runs: number of runs
    err_func: function used for error computation

  '''

  sens_types = ['no2', 'o3']

  terms_incl = [[False, False],
                [True, False],
                [False, True],
                [True, True]]

  tree_params = [(2, 50),
                 (3, 50),
                 (3, 150),
                 (4, 50)]

  # need to change the 12 and the 4 in future iterations
  errs = np.zeros([len(dfs), len(grans), 12, runs, len(sens_types), 4, 4])

  def _run_iters(X, y, reg_func, loc, **kwargs):

    def _run_reg(X, y, **kwargs):
      ret = reg_func(X, None, y, **kwargs)
      if isinstance(ret, tuple):
        metrics = ret[-1]
      else:
        metrics = ret
      return metrics[:4]

    for run in xrange(runs):
      for l, X_ in enumerate(X):
        errs[loc[0], loc[1], loc[2], run, loc[3], l, :] = _run_reg(X_, y, **kwargs)

      for l in xrange(len(X)):
        agg_X, agg_y, labels, dummy1, dummy2 = get_agg_sets(X, y, l)
        kwargs['labels'] = labels
        errs[loc[0], loc[1], loc[2], run, loc[3], l+2, :] =\
                   _run_reg(agg_X, agg_y, **kwargs)


  # alphasense values do not train on aggregate models
  errs[:, :, :4, :, :, 2:] = np.inf

  for (i, df) in enumerate(dfs):
    if i == 1:
      grans_ = [60]
    else:
      grans_ = grans
    for (j, gran) in enumerate(grans_):
      # average data
      df = window_avg(df, gran).dropna()

      # errors for alphasense models index 0 to 3
      print "For alphasense: (granularity = %dmins)" % gran
      for run in xrange(runs):

        vals = df.values
        vals = np.random.permutation(vals)[:30]
        err_no2, err_o3 = alp.alphasense_compute(vals, True, True)

        errs[i, j, :4, run, 0, :2, :] = np.swapaxes(err_no2[:, :, :4], 0, 1)
        errs[i, j, :4, run, 1, :2, :] = np.swapaxes(err_o3[:, :, :4], 0, 1)
      
      # errors for different linear models
      #print "For linear models: (granularity = %dmins)" % gran
      #for k, term in enumerate(terms_incl):
      #  epochs, no2_x, no2_y, ox_x, o3_y, ox_y, coeffs_no2_names,\
      #  coeffs_ox_names = preproc(df, temps_present=True, hum_present=True,
      #        incl_op1=True, incl_op2=True, incl_temps=term[0], incl_op1t=False,
      #        incl_op2t=False, incl_hum=term[0], incl_op1h=False, incl_op2h=False,
      #        incl_op12=False, incl_cross_terms=term[1], clean=3,
      #        runs=200, loc_label='', training_set_ratio=0.7)
#
#        _run_iters(no2_x, no2_y, regress_once, (i, j, k+4, 0))
#        _run_iters(ox_x, o3_y, regress_once, (i, j, k+4, 1))

      # errors for model trees
      #print "For model trees: (granularity = %dmins)" % gran
      #epochs, no2_x, no2_y, ox_x, o3_y, ox_y, coeffs_no2_names,\
      #coeffs_ox_names = preproc(df, temps_present=True, hum_present=True,
      #      incl_op1=True, incl_op2=True, incl_temps=False, incl_op1t=False,
      #      incl_op2t=False, incl_hum=False, incl_op1h=False, incl_op2h=False,
      #      incl_op12=False, incl_cross_terms=True, clean=3,
      #      runs=200, loc_label='', training_set_ratio=0.7)
#
#      for k, param in enumerate(tree_params):
#        print "\tFor tree: ", param
#        _run_iters(no2_x, no2_y, model_tree_regress, (i, j, k+8, 0),
#                   max_depth=param[0], min_samples_leaf=param[1])
#        _run_iters(ox_x, o3_y, model_tree_regress, (i, j, k+8, 1),
#                   max_depth=param[0], min_samples_leaf=param[1])
#
  np.save('global_errs_corr_as.npy', errs)
  return errs
  
# ----------------------------------------------------------------------------------
def mape_plots_vec(err_vec):
  '''
    Get MAPE error plots for SATVAM eScience 2019 paper from DataFrame

    err_vec: of shape [Loc x Gran x model x runs x senstype x test-set x metrics]
  '''

  ID_LOC = 0
  ID_GRAN = 1
  ID_METRIC = 2  # 0:mae, 1:rmse, 2:mape, 3:r2

  MOD_BEST = [0, 6, 10]

  ret_figs = []
  ret_names = []

  for i in xrange(err_vec.shape[4]):
    print "\n Sensor type %d" % i
    for j in range(2, err_vec.shape[5]):

      if i == 1 and j == 2:
        scale = [0, 200]
        leg = True
      else:
        if i == 1:
          scale = [0, 200]
        else:
          scale = [0, 150]
        leg = False

      '''
      print "\n Test set: %d" % j
      print "Effect of models...."
      # effect of models
      mapes = err_vec[ID_LOC, ID_GRAN, :, :, i, j, ID_METRIC].T
      print mapes.shape
      fig, ax = plotting.plot_violin(mapes,
                  #title=r'\textit{Effect of models on MAPE}',
                  xlabel='', ylabel=r'$ MAPE (\%) $',
                  x_tick_labels=[['A1', 'A2', 'A3', 'A4'],
                                 ['L1', 'L2', 'L3', 'L4'],
                                 ['R1', 'R2', 'R3', 'R4']],
                  groups = [4, 4, 4], scale=[0, 100], leg=leg,
                  group_labels=['AlphaSense', 'Lin Regr', 'Regr Tree'])

      ret_figs.append(fig)
      ret_names.append('violins_effmodels_type%d_set%d' % (i ,j))

      print "Effect of granularity...."
      # effect of granularity
      mapes = err_vec[ID_LOC, :, MOD_BEST, :, i, j, ID_METRIC]  # M x G x R
      mapes = np.swapaxes(mapes, 0, 1)
      print mapes.shape
      mapes = np.reshape(mapes, [mapes.shape[0] *
                                 mapes.shape[1], mapes.shape[2]], order='F').T

      fig, ax = plotting.plot_violin(mapes,
                  #title=r'\textit{Effect of granularity on MAPE}',
                  xlabel='', ylabel=r'$ MAPE (\%) $',
                  x_tick_labels=['1', '15', '60'],
                  groups = [3, 3, 3], scale=scale,
                  group_labels=['A1', 'L3',
                                r'R3'], leg=leg,
                  mark_maxoutliers=False)

      ret_figs.append(fig)
      ret_names.append('violins_effgrans_type%d_set%d' % (i ,j))

      #if i != 1 or j != 3:
      #  continue
        
      print "Effect of location...."
      # effect of location
      mapes = err_vec[:, 2, MOD_BEST, :, i, j, ID_METRIC]  # M x L x R
      mapes = np.swapaxes(mapes, 0, 1)
      print mapes.shape
      mapes = np.reshape(mapes, [mapes.shape[0] *
                                 mapes.shape[1], mapes.shape[2]], order='F').T

      fig, ax = plotting.plot_violin(mapes,
                  #title=r'\textit{Effect of location on MAPE}',
                  xlabel='', ylabel=r'$ MAPE (\%) $',
                  x_tick_labels=['DEL', 'MUM'],
                  groups = [2, 2, 2], scale=[0, 200],
                  group_labels=['A1', 'L3',
                                r'R3'], leg=leg)

      ret_figs.append(fig)
      ret_names.append('violins_effloc_type%d_set%d' % (i, j))
    '''
  
    print "Effect of sensors...."
    # effect of sensors
    mapes = err_vec[ID_LOC, ID_GRAN, MOD_BEST, :, i, :, ID_METRIC] # M x R x T
    mapes = np.swapaxes(mapes, 1, 2)
    mapes = np.swapaxes(mapes, 0, 1)
    print mapes.shape
    mapes = np.reshape(mapes, [mapes.shape[0] *
                               mapes.shape[1], mapes.shape[2]], order='F').T

    if i == 1:
      scale = [0, 200]
    else:
      scale = [0, 110]

    fig, ax = plotting.plot_violin(mapes,
                  #title=r'\textit{Effect of sensors on MAPE}',
                  xlabel='', ylabel=r'$ MAPE (\%) $',
                  x_tick_labels=['1:1', '2:2', 'X:1', 'X:2'],
                  groups = [4, 4, 4], scale=scale,
                  group_labels=['AlphaSense A1', 'Lin Regr. L3',
                                r'Regr Tree R3'], leg=False, mark_maxoutliers=False)

    ret_figs.append(fig)
    ret_names.append('violins_effsens_type%d' % i)

  return ret_figs, ret_names

# ----------------------------------------------------------------------------------
def pm_correlate(data, ref_pm1_incl=False, ref_pm10_incl=False, loc_label='---'):
  
  # list of all figures plotted
  figs = []
  fignames = []

  # remove sub-zero ppb values
  data = data[data.applymap(lambda x: x > 0).all(1)]

  ts = data.values[:,0]
  pm1_vals = []
  pm25_vals = []
  pm10_vals = []

  leg_labels = []

  start_i = 2
  if ref_pm1_incl:
    start_i=3
    pm1_vals.append(data.values[:, 1])
    pm25_vals.append(data.values[:, 2])
    if ref_pm10_incl:
      start_i=4
      pm10_vals.append(data.values[:, 3])

  else:
    pm25_vals.append(data.values[:, 1])
    if ref_pm10_incl:
      start_i=3
      pm10_vals.append(data.values[:, 2])

  for i in range(start_i, np.size(data.values, 1), 3):
    pm1_vals.append(data.values[:, i].tolist())
    pm25_vals.append(data.values[:, i+1].tolist())
    pm10_vals.append(data.values[:, i+2].tolist())

    leg_labels.append('Sensor %d' % int(((i-2)/3) + 1))

  leg_labels_pm1 = leg_labels[:]
  leg_labels_pm10 = leg_labels[:]
  leg_labels.insert(0, 'Reference')

  # TODO: Minute average PM1 and PM10 data
  pm1_vals = np.array(pm1_vals).T
  pm25_vals = np.array(pm25_vals).T
  pm10_vals = np.array(pm10_vals).T

  if np.shape(pm1_vals)[1] < 2:
    leg_labels_pm1.insert(0, 'Reference')

  if np.shape(pm10_vals)[1] < 2:
    leg_labels_pm10.insert(0, 'Reference')

  fig, ax = plotting.ts_plot(ts, pm1_vals,
    title = r'$ PM_{1.0} $ concentration',
    ylabel = r'\textit{Concentration ($ ug/m^3 $)}',
    leg_labels=leg_labels_pm1)

  for i in range(1, np.size(pm25_vals, axis=1)):
    text = get_corr_txt(pm25_vals[:, i].astype(float),
        pm25_vals[:, 0].astype(float), add_title='S%d' % i)

    x = i / 5.0
    ax.annotate(text, xy = (x, 0.75), xycoords='axes fraction')

  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy=(0.6, 0.3), xycoords='axes fraction')

  figs.append(fig)
  fignames.append('pm1-comp')

  fig, ax = plotting.ts_plot(ts, pm10_vals,
    title = r'$ PM_{10} $ concentration',
    ylabel = r'\textit{Concentration ($ ug/m^3 $)}',
    leg_labels=leg_labels_pm10)

  for i in range(1, np.size(pm25_vals, axis=1)):
    text = get_corr_txt(pm25_vals[:, i].astype(float),
        pm25_vals[:, 0].astype(float), add_title='S%d' % i)

    x = i / 5.0
    ax.annotate(text, xy = (x, 0.75), xycoords='axes fraction')

  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy=(0.6, 0.3), xycoords='axes fraction')

  figs.append(fig)
  fignames.append('pm10-comp')

  fig, ax = plotting.ts_plot(ts, pm25_vals,
    title = r'$ PM_{2.5} $ concentration',
    ylabel = r'\textit{Concentration ($ ug/m^3 $)}',
    leg_labels=leg_labels)

  for i in range(1, np.size(pm25_vals, axis=1)):
    text = get_corr_txt(pm25_vals[:, i].astype(float),
        pm25_vals[:, 0].astype(float), add_title='S%d' % i)

    x = i / 5.0
    ax.annotate(text, xy = (x, 0.75), xycoords='axes fraction')
  
  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy=(0.6, 0.3), xycoords='axes fraction')

  figs.append(fig)
  fignames.append('pm25-comp')

  return figs, fignames
# ----------------------------------------------------------------------------------
