# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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
import stats
# ------------------------------------------------------------------------------
CONF_DECIMALS = 6
# ------------------------------------------------------------------------------
# list of all figures plotted
no2_figs = []
o3_figs = []
no2_fignames = []
o3_fignames = []
# ==============================================================================
# Statistics routines
def clean_data(X, sigma_mult):
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

  X = X.replace([np.inf, -np.inf], np.nan).dropna()
  sizex = len(X)
  # remove non-positive ppb values
  X = X[X.applymap(lambda x: x > 0).all(1)]

  # remove values beyond +-sigma_mult * sigma
  mu = np.mean(X.values, axis=0)
  sigma = np.std(X.values, axis=0)
  
  ranges = [(mu - sigma_mult * sigma).tolist(),
            (mu + sigma_mult * sigma).tolist()]

  for i in xrange(1, X.values.shape[1]):
    X = X[X.iloc[:,[i]].applymap(lambda x: (x > ranges[0][i]
         and x < ranges[1][i])).all(1)]

  print "%d entries dropped" % (sizex - len(X))

  return X
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

  fig = plotting.inset_hist_fig(fig, ax, no2_y, ['25%', '25%'], 1, ids=[0])
  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')

  no2_figs.append(fig)
  no2_fignames.append('no2-ref')

  fig, ax = plotting.ts_plot(epochs, o3_y,
         title=r'$ O_3 $ \textbf{readings from Reference monitor}',
         ylabel=r'$ O_3 $\textit{concentration (ppb)}', ylim=[0, 100],
         leg_labels=['$ O_3 $ conc (ppb)'], ids=[0])

  fig = plotting.inset_hist_fig(fig, ax, o3_y, ['25%', '25%'], 1, ids=[0])
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

  fig = plotting.inset_hist_fig(fig, ax, no2_op1_vals,
                                ['25%', '25%'], 1, ids=ids)
  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')

  no2_figs.append(fig)
  no2_fignames.append('no2-op1')

  fig, ax = plotting.ts_plot(epochs, no2_op2_vals,
         title=r'$ NO_2 $ \textbf{Output2 from AlphaSense sensors}',
         ylabel=r'$ NO_2 $ \textit{op2 (mV)}', ylim=[210, 250],
         leg_labels=[('Sensor %d' % x) for x in range(1, len(no2_x)+1)],
         ids=ids)

  fig = plotting.inset_hist_fig(fig, ax, no2_op2_vals,
                                 ['25%', '25%'], 1, ids=ids)
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

  fig = plotting.inset_hist_fig(fig, ax, ox_op1_vals,
                                ['25%', '25%'], 1, ids=ids)
  #txt = watermark(ax, loc_label, '')
  #ax.annotate(txt, xy = (0.6, 0.3), xycoords='axes fraction')

  o3_figs.append(fig)
  o3_fignames.append('o3-op1')

  fig, ax = plotting.ts_plot(epochs, ox_op2_vals,
         title=r'$ OX (NO_2 + O_3) $ \textbf{Output 2 from AlphaSense sensors}',
         ylabel=r'$ OX $ \textit{op2 (mV)}', ylim=[210, 250],
         leg_labels=[('Sensor %d' % x) for x in range(1, len(ox_x)+1)],
         ids=ids)

  fig = plotting.inset_hist_fig(fig, ax, ox_op2_vals,
                                ['25%', '25%'], 1, ids=ids)
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
  tick_labs.append('Train Agg')

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
# ==============================================================================
# Routines for regression
def regress_once(X, y, labels=None, train_size=0.7, intercept=True):
  '''
     Perform regression on the given data-set and return coefficients
     and error metrics
    
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

  X_ = X[perm, :]
  y_ = y[perm]

  X_train = X_[:train_size, :]
  y_train = y_[:train_size]
  X_test = X_[train_size:, :]
  y_test = y_[train_size:]

  # train model
  reg = LinearRegression(fit_intercept=intercept).fit(X_train, y_train)
    
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

  # training model
  pred = reg.predict(X_train)
  dev = stats.mae(y_train, pred)

  coeffs = reg.coef_.tolist()
  coeffs.append(reg.intercept_)

  return coeffs, metrics
# ------------------------------------------------------------------------------
def regress_sensor_type(X, y, train_ratio, runs):
  '''
     Train on all sensors of a certain type and obtain self
     trained and cross trained error measures

     Params:
      X       - list of training inputs for each sensor
      y       - training output values common to all
      train_ratio - Ratio of training set to consider
      runs    - Number of runs for each sensor

     Return:
      coeffs, maes, rmses, mapes

      coeffs has shape [num_sensors, runs, (num_sensors + 1), num_coeffs]
      maes, rmses and mapes have shape [num_sensors, runs, (num_sensors + 1)]
  '''

  # coefficients for all steps carried out
  # shape: N * R * (N + 1) * m
  coeffs = np.zeros([len(X), runs, (len(X) + 1), (1+np.shape(X[0])[1])])

  # error metrics
  # shape: N * R * (N + 1)
  maes = np.zeros([len(X), runs, (len(X) + 1)])
  rmses = np.zeros([len(X), runs, (len(X) + 1)]) 
  mapes = np.zeros([len(X), runs, (len(X) + 1)]) 

  for j in xrange(len(X)):

    print "Sensor: " + str(j + 1)
    for i in xrange(runs):

      sys.stdout.write("\rRun ........ %d (%.2f%% complete)"
                            % ((i+1), (i+1) * 100/runs))
      for k in xrange(len(X)):

        # Train on sensor k
        tmpcoeffs, metrics = regress_once(X[k], y, train_size=train_ratio)
        coeffs[j, i, k][:] = tmpcoeffs
        maes[j, i, k] = metrics[0]
        rmses[j, i, k] = metrics[1]
        mapes[j, i, k] = metrics[2]

      # build aggregate dataset
      agg_X = []
      agg_y = np.array([])
      labels = np.zeros([len(X) * np.shape(X[0])[0],])
      for (k, X_sensor) in enumerate(X):

        agg_y = np.concatenate((agg_y, y), axis=0)
        if k == 0:
          agg_X = X_sensor
        else:
          agg_X = np.concatenate((agg_X, X_sensor), axis=0)

        if k == j:
          labels[k * np.shape(X[0])[0] : (k+1) * np.shape(X[0])[0]] = 1

      # train on aggregate dataset
      tmpcoeffs, metrics = regress_once(agg_X, agg_y,
                    labels=labels, train_size=train_ratio)
      coeffs[j, i, -1][:] = tmpcoeffs
      maes[j, i, -1] = metrics[0]
      rmses[j, i, -1] = metrics[1]
      mapes[j, i, -1] = metrics[2]

    print ""

    # visualize predictions
    #if temps_present:
    #  plot_ts_prediction(epochs, no2_y, predict_no2, o3_y,
    #                     predict_o3, j, comp_witht=temps_present, temps=temp[j])
    #else:
    #  plot_ts_prediction(epochs, no2_y, predict_no2, o3_y, predict_o3, j)
    
    print "Sensor %d DONE" % (j+1)

  return coeffs, maes, rmses, mapes

# ------------------------------------------------------------------------------
def regress_df(data, temps_present=False, hum_present=False,
               incl_temps=False, incl_op2t=False,
               incl_hum=False, incl_op2h=False,
               clean=3, runs=1000, loc_label='---'):

  # remove outliers
  data = clean_data(data, clean)

  # remove possible outliers
  #tmp_data = pd.DataFrame(data.iloc[:, 0])
  #data = data[tmp_data.applymap(lambda x: x < 200).all(1)]

  training_set_ratio = 0.7

  train_size = int(np.floor(training_set_ratio * data.shape[0]))
  print "Training set size: \t", train_size
  print "Test set size: \t", (np.size(data, 0) - train_size)

  no2_x = []
  ox_x = []
  temp = []
  hum = []

  no2_y_pred = []
  o3_y_pred = []

  coeffs_no2_names = ['op1', 'op2']
  coeffs_ox_names = ['op1', 'op2']

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

  epochs = data.values[:,0]

  # store x and y values
  no2_y = data.values[:,1]
  o3_y = data.values[:,2]

  # convert o3 to ox for regression
  ox_y = o3_y + no2_y

  if temps_present:
    if incl_temps:
      coeffs_no2_names.append('temp')
      coeffs_ox_names.append('temp')

    if incl_op2t:
      coeffs_no2_names.append('op2T')
      coeffs_ox_names.append('op2T')

  if hum_present:
    if incl_hum:
      coeffs_no2_names.append('hum')
      coeffs_ox_names.append('hum')

    if incl_op2h:
      coeffs_no2_names.append('op2h')
      coeffs_ox_names.append('op2h')
      
  coeffs_no2_names.append('constant')
  coeffs_ox_names.append('constant')
    
  for i in xrange(np.size(data.values, 1)):
    if col_ox(i)[-1] >= np.size(data.values, 1):
      break

    tmp_idx_n = col_no2(i)
    tmp_idx_o = col_ox(i)

    if temps_present:
      temp.append(data.values[:,col_temp(i)])
      if incl_temps:
        tmp_idx_n.append(col_temp(i))
        tmp_idx_o.append(col_temp(i))

    if hum_present:
      hum.append(data.values[:,col_hum(i)])
      if incl_hum:
        tmp_idx_n.append(col_hum(i))
        tmp_idx_o.append(col_hum(i))

    no2_x.append(data.values[:,tmp_idx_n])
    ox_x.append(data.values[:,tmp_idx_o])

    if temps_present and incl_op2t:
      no2_op2t = np.reshape(data.values[:, col_no2(i)[1]]
        * data.values[:, col_temp(i)], [np.shape(data.values)[0], 1])
      ox_op2t = np.reshape(data.values[:, col_ox(i)[1]]
        * data.values[:, col_temp(i)], [np.shape(data.values)[0], 1])

      no2_x[i] = np.concatenate((no2_x[i], no2_op2t), axis=1)
      ox_x[i] = np.concatenate((ox_x[i], ox_op2t), axis=1)

    if hum_present and incl_op2h:
      no2_op2h = np.reshape(data.values[:, col_no2(i)[1]]
        * data.values[:, col_hum(i)], [np.shape(data.values)[0], 1])
      ox_op2h = np.reshape(data.values[:, col_ox(i)[1]]
        * data.values[:, col_hum(i)], [np.shape(data.values)[0], 1])

      no2_x[i] = np.concatenate((no2_x[i], no2_op2h), axis=1)
      ox_x[i] = np.concatenate((ox_x[i], ox_op2h), axis=1)

  #visualize_rawdata(epochs, no2_x, no2_y, ox_x, o3_y)

  # process and regress data: multifold
  print "\nFor NO_2 sensors....."
  coeffs_no2, maes_no2, rmses_no2, mapes_no2 = regress_sensor_type(
        no2_x, no2_y, training_set_ratio, runs)

  print "\nFor OX sensors....."
  coeffs_ox, maes_ox, rmses_ox, mapes_ox = regress_sensor_type(
        ox_x, ox_y, training_set_ratio, runs)
  
  # mean coefficients should have the shape (N+1) * m
  mean_no2_coeffs = np.mean(coeffs_no2, axis=(0, 1)).T
  mean_ox_coeffs = np.mean(coeffs_ox, axis=(0, 1)).T

  # plot violins for error metrics
  plot_error_violins(maes_no2, name='MAE',
        full_name='Mean Absolute Error', sens_type='NO_2');
  plot_error_violins(rmses_no2, name='RMSE',
        full_name='Root Mean Square Error', sens_type='NO_2');
  plot_error_violins(mapes_no2, name='MAPE',
        full_name='Mean Absolute Percentage Error', sens_type='NO_2');

  plot_error_violins(maes_ox, name='MAE',
        full_name='Mean Absolute Error', sens_type='O_X');
  plot_error_violins(rmses_ox, name='RMSE',
        full_name='Root Mean Square Error', sens_type='O_X');
  plot_error_violins(mapes_ox, name='MAPE',
        full_name='Mean Absolute Percentage Error', sens_type='O_X');

  return no2_figs, no2_fignames, o3_figs, o3_fignames

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
