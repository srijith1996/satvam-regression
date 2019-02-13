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

import plotting
import stats
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
  r2 = stats.coeff_deter(y_true, y_pred)

  text = r'\textbf{Correlation Stats %s}'% add_title
  text = text + '\n' + r'$ RMSE = %0.3f $' % rmse
  text = text + '\n' + r'$ MAPE = %0.3f $' % mape
  text = text + '\n' + r'$ MAE  = %0.3f $' % mae
  text = text + '\n' + r'$ r_P  = %0.3f $' % pearson
  text = text + '\n' + r'$ R^2  = %0.3f $' % r2

  return text
# ------------------------------------------------------------------------------
def regress_once(X, y, train_size=0.7, intercept=True):
  
  train_size = int(np.floor(train_size * X.shape[0]))

  perm = np.arange(X.shape[0])
  perm = np.random.permutation(perm)

  X_ = X[perm, :]
  y_ = y[perm]

  X_train = X_[:train_size, :]
  y_train = y_[:train_size]

  X_test = X_[train_size:, :]
  y_test = y_[train_size:]

  # TODO: subtracting zero offsets
  #no2_x_train[0] -= we_zero_offsets[j]
  #no2_x_train[1] -= ae_zero_offsets[j]
    
  # train model
  reg = LinearRegression(fit_intercept=intercept).fit(X_train, y_train)
    
  # TODO: Publish to report
  #print "Regression coefficients: \t", reg_no2.coef_
  #print "Regression intercept: \t", reg_no2.intercept_
  #print "coeff of AE: \t", -(reg_no2.coef_[1] /reg_no2.coef_[0])
  #print "sensitivity: \t", 1/reg_no2.coef_[0]
  
  # test model
  pred = reg.predict(X_test)
  dev = stats.mae(y_test, pred)
  #print "Deviation measure L2 test set: \t", dev

  # training model
  pred = reg.predict(X_train)
  dev = stats.mae(y_train, pred)
  #print "Deviation measure L2 training set: \t", dev
      
  #print "R^2 training set: \t", reg.score(X_train, y_train)
  #print "R^2 test set: \t", reg.score(X_test, y_test)
    
  predict = reg.predict(X)
  residue = y - predict
  coeffs = reg.coef_.tolist()
  coeffs.append(reg.intercept_)

  return coeffs, predict, residue
# ------------------------------------------------------------------------------
def clean_data(X, sigma_mult):
  
  print "Cleaning x+-%dsigma" % sigma_mult

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

  X = X.replace([np.inf, -np.inf], np.nan).dropna()
  print "%d entries dropped" % (sizex - len(X))

  return X
# ------------------------------------------------------------------------------
def regress_df(data, temps_present=False,
               incl_temps=False, incl_op2t=False,
               clean=3, runs = 1000):

  # list of all figures plotted
  no2_figs = []
  o3_figs = []

  no2_fignames = []
  o3_fignames = []
  
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
  temp = []
  ox_x = []

  no2_y_pred = []
  o3_y_pred = []

  coeffs_no2_names = ['op1', 'op2']
  coeffs_ox_names = ['op1', 'op2']

  # column locations for no2, ox and temperature data of the ith sensor
  col_skip = 3

  if temps_present:
    col_no2 = (lambda i: range((col_skip + 5*i + 1),(col_skip + 5*i + 3)))
    col_ox = (lambda i: range((col_skip + 5*i + 3),(col_skip + 5*i + 5)))
    col_temp = (lambda i: (col_skip + 5*i))
  else:
    col_no2 = (lambda i: range((col_skip + 4*i),(col_skip + 4*i + 2)))
    col_ox = (lambda i: range((col_skip + 4*i + 2),(col_skip + 4*i + 4)))

  epochs = data.values[:,0]

  # store x and y values
  no2_y = data.values[:,1]
  o3_y = data.values[:,2]

  if temps_present and incl_temps:
    coeffs_no2_names.append('temp')
    coeffs_ox_names.append('temp')

  if temps_present and incl_op2t:
    coeffs_no2_names.append('op2T')
    coeffs_ox_names.append('op2T')

  coeffs_no2_names.append('constant')
  coeffs_ox_names.append('constant')
    
  for i in xrange(np.size(data.values, 1)):
    if col_ox(i)[1] >= np.size(data.values, 1):
      break

    tmp_idx_n = col_no2(i)
    tmp_idx_o = col_ox(i)

    if temps_present:
      temp.append(data.values[:,col_temp(i)])
      if incl_temps:
        tmp_idx_n.append(col_temp(i))
        tmp_idx_o.append(col_temp(i))

    no2_x.append(data.values[:,tmp_idx_n])
    ox_x.append(data.values[:,tmp_idx_o])

    if temps_present and incl_op2t:
      no2_op2t = np.reshape(data.values[:, col_no2(i)[1]]
        * data.values[:, col_temp(i)], [np.shape(data.values)[0], 1])
      ox_op2t = np.reshape(data.values[:, col_ox(i)[1]]
        * data.values[:, col_temp(i)], [np.shape(data.values)[0], 1])

      no2_x[i] = np.concatenate((no2_x[i], no2_op2t), axis=1)
      ox_x[i] = np.concatenate((ox_x[i], ox_op2t), axis=1)

  # convert o3 to ox for regression
  ox_y = o3_y + no2_y

  # ---------------------------- VISUALIZATION ------------------------------
  # visualize time-series of ref data
  fig, ax = plotting.ts_plot(epochs, no2_y,
         title=r'$ NO_2 $\textbf{ readings from Reference monitor}',
         ylabel=r'$ NO_2 $\textit{ concentration (ppb)}',
         leg_labels=[r'$ NO_2 $ conc (ppb)'])

  fig = plotting.inset_hist_fig(fig, ax, no2_y, ['35%', '35%'], 1)

  no2_figs.append(fig)
  no2_fignames.append('no2-ref.svg')

  fig, ax = plotting.ts_plot(epochs, o3_y,
         title=r'$ O_3 $ \textbf{readings from Reference monitor}',
         ylabel=r'$ O_3 $\textit{concentration (ppb)}',
         leg_labels=['$ O_3 $ conc (ppb)'])

  fig = plotting.inset_hist_fig(fig, ax, o3_y, ['35%', '35%'], 1)

  o3_figs.append(fig)
  o3_fignames.append('o3-ref.svg')

  # visualize time-series of AlphaSense sensors
  # TODO: Change labels based on sensor name 
  no2_op1_vals = np.zeros([np.shape(no2_x[0])[0], len(no2_x)])
  no2_op2_vals = np.zeros([np.shape(no2_x[0])[0], len(no2_x)])
  ox_op1_vals = np.zeros([np.shape(ox_x[0])[0], len(ox_x)])
  ox_op2_vals = np.zeros([np.shape(ox_x[0])[0], len(ox_x)])

  for (i, sens_no2) in enumerate(no2_x):
    no2_op1_vals[:, i] = sens_no2[:, 0]
    no2_op2_vals[:, i] = sens_no2[:, 1]

  fig, ax = plotting.ts_plot(epochs, no2_op1_vals,
         title=r'$ NO_2 $ \textbf{Output1 from AlphaSense sensors}',
         ylabel=r'$ NO_2 $ \textit{op1 (mV)}', ylim=[190, 300],
         leg_labels=[('Sensor %d' % x) for x in range(1, len(no2_x)+1)])

  fig = plotting.inset_hist_fig(fig, ax, no2_op1_vals, ['35%', '35%'], 1)

  no2_figs.append(fig)
  no2_fignames.append('no2-op1.svg')

  fig, ax = plotting.ts_plot(epochs, no2_op2_vals,
         title=r'$ NO_2 $ \textbf{Output2 from AlphaSense sensors}',
         ylabel=r'$ NO_2 $ \textit{op2 (mV)}', ylim=[190, 300],
         leg_labels=[('Sensor %d' % x) for x in range(1, len(no2_x)+1)])

  fig = plotting.inset_hist_fig(fig, ax, no2_op2_vals, ['35%', '35%'], 1)

  no2_figs.append(fig)
  no2_fignames.append('no2-op2.svg')

  for (i, sens_ox) in enumerate(ox_x):
    ox_op1_vals[:, i] = sens_ox[:, 0]
    ox_op2_vals[:, i] = sens_ox[:, 1]

  fig, ax = plotting.ts_plot(epochs, ox_op1_vals,
         title=r'$ OX (NO_2 + O_3) $ \textbf{Output 1 from AlphaSense sensors}',
         ylabel=r'$ OX $ \textit{op1 (mV)}',# ylim=[190, 300],
         leg_labels=[('Sensor %d' % x) for x in range(1, len(ox_x)+1)])

  fig = plotting.inset_hist_fig(fig, ax, ox_op1_vals, ['35%', '35%'], 1)

  o3_figs.append(fig)
  o3_fignames.append('o3-op1.svg')

  fig, ax = plotting.ts_plot(epochs, ox_op2_vals,
         title=r'$ OX (NO_2 + O_3) $ \textbf{Output 2 from AlphaSense sensors}',
         ylabel=r'$ OX $ \textit{op2 (mV)}',# ylim=[190, 300],
         leg_labels=[('Sensor %d' % x) for x in range(1, len(ox_x)+1)])

  fig = plotting.inset_hist_fig(fig, ax, ox_op2_vals, ['35%', '35%'], 1)

  o3_figs.append(fig)
  o3_fignames.append('o3-op2.svg')
  # -------------------------------------------------------------------------

  # ---------------------------MULTIFOLD REGRESSION--------------------------
  # process and regress data: multifold

  # TODO
  #we_zero_offsets = [227, 229, 222]
  #ae_zero_offsets = [228, 230, 221]

  resid_no2 = np.zeros([len(no2_y), 1])
  resid_ox = np.zeros([len(ox_y), 1])
  coeffs_no2 = []
  coeffs_ox = []

  predict_no2 = []
  predict_o3 = []
  
  for j in xrange(len(no2_x)):
    print "Sensor: " + str(j + 1)
    coeffs_no2.append([])
    coeffs_ox.append([])
  
    for i in xrange(runs):

      sys.stdout.write("\rEpoch ............ %d" % (i+1))

      coeffs, predict_no2, resid_no2 = regress_once(no2_x[j],
          no2_y, training_set_ratio)
      
      coeffs_no2[j].append(coeffs)
    
      coeffs, predict, resid_ox = regress_once(ox_x[j],
          ox_y, training_set_ratio)
      
      coeffs_ox[j].append(coeffs)
      predict_o3 = predict - predict_no2

    print "\n"

    no2_y_pred.append(predict_no2)
    o3_y_pred.append(predict_o3)
  # -------------------------------------------------------------------------

  # ---------------------------- VISUALIZATION ------------------------------
    # plot predicted vs. true ppb
    print "plotting actual and predicted values: NO2"
    t_series = np.array([no2_y, predict_no2]).T

    fig, ax = plotting.ts_plot(epochs, t_series,
        title = r'\textbf{True and Predicted concentrations of }'
              + r'$ NO_2 $ \textbf{ (Sensor %d)}' % (j + 1),
        ylabel = r'Concentration (ppb)',
        leg_labels=['Reference conc.', 'Predicted conc.'])

    text = get_corr_txt(t_series[:, 0], t_series[:, 1])
    ax.annotate(text, xy = (0.7, 0.75), xycoords='axes fraction')

    no2_figs.append(fig)
    no2_fignames.append('no2-sens%d-predict-true-comp.svg' % (j+1))

    print "plotting actual and predicted values: O3"
    t_series = np.array([o3_y, predict_o3]).T

    fig, ax = plotting.ts_plot(epochs, t_series,
        title = r'\textbf{True and Predicted concentrations of } $ O_3 $',
        ylabel = r'Concentration (ppb)',
        leg_labels=['Reference conc.', 'Predicted conc.'])

    text = get_corr_txt(t_series[:, 0], t_series[:, 1])
    ax.annotate(text, xy = (0.7, 0.75), xycoords='axes fraction')

    o3_figs.append(fig)
    o3_fignames.append('o3-sens%d-predict-true-comp.svg' % (j+1))

    # plot regression surface
    #print "plotting regression surface"
    #fig = plotting.plot_reg_plane(no2_x_train, no2_y_train,
    #                 [reg_no2.intercept_, reg_no2.coef_[0], reg_no2.coef_[1]],
    #                 title=r'\textbf{Outputs vs. ppb on training set}',
    #                 xlabel=r'\textbf{sensor output WE (mV)}',
    #                 ylabel=r'\textbf{sensor output AE (mV)}',
    #                 zlabel=r'$ NO_2 $\textit{concentration (ppb)}')
    #figs.append(fig)

    #fig = plotting.plot_reg_plane(no2_x_test, no2_y_test,
    #                 [reg_no2.intercept_, reg_no2.coef_[0], reg_no2.coef_[1]],
    #                 title=r'\textbf{Outputs vs. ppb on test set}',
    #                 xlabel=r'\textbf{sensor output WE (mV)}',
    #                 ylabel=r'\textbf{sensor output AE (mV)}',
    #                 zlabel=r'$ NO_2 $\textit{concentration (ppb)}')
    #figs.append(fig)


    # plot residuals wrt time
    print "plotting residual characteristics"
    ylim_p = [-150, 50]
    ylim_s = [0, 45]
    
    fig_n = None
    fig_o = None
    if not temps_present:
      fig_n, ax = plotting.ts_plot(epochs, resid_no2,
            title=r"\textbf{Prediction errors (} $ NO_2 $ \textbf{) vs temperature (Sensor "
                + str(j + 1) + ")}",
            ylabel=r"\textit{Residuals (ppb)}", ylim=ylim_p,
            leg_labels=["Residual error"])

      fig_o, ax = plotting.ts_plot(epochs, resid_o3,
            title=r"\textbf{Prediction errors (} $ O_3 $ \textbf{) vs temperature (Sensor "
                + str(j + 1) + ")}",
            ylabel=r"\textit{Residuals (ppb)}", ylim=ylim_p,
            leg_labels=["Residual error"])

    else:
      fig_n, ax = plotting.compare_ts_plot(epochs, resid_no2, temp[j],
            title=r"\textbf{Prediction errors (} $ NO_2 $ \textbf{) vs temperature (Sensor "
               + str(j + 1) + ")}",
            ylabel=r"\textit{Residuals (ppb)}",
            ylabel_s=r"\textit{Temperature} ($ ^{\circ} C $)", ylim_p=ylim_p,
            ylim_s=ylim_s, leg_labels=["Residual error", "Temperature"])

      # compute r^2 between residual and temperature
      p = np.polyfit(temp[j].astype(float), resid_no2.astype(float), 1)
      r2 = stats.coeff_deter(temp[j].astype(float), resid_no2.astype(float))
  
      plot_str = "$ e = %.3f * T + %.3f $" % (p[0], p[1])
      plot_str2 = "$ {R^2}_{eT} = %.4f $" % r2
      ax.text((epochs[-1] + 3*epochs[0])/4, (ylim_p[0] + 19*ylim_p[1])/20,
              plot_str, ha="center", va="bottom")
      ax.text((epochs[-1] + 3*epochs[0])/4, (ylim_p[0] + 19*ylim_p[1])/20,
              plot_str2, ha="center", va="top")

      ax.annotate("$ e = y_{pred} - y_{true} $", xy=(0.7, 0.9),
                  xycoords="axes fraction")
    
      # for O3
      fig_o, ax = plotting.compare_ts_plot(epochs, resid_o3[:, 0], temp[j],
            title=r"\textbf{Prediction errors (} $ O_3 $ \textbf{) vs temperature (Sensor "
                + str(j + 1) + ")}",
            ylabel=r"\textit{Residuals (ppb)}",
            ylabel_s=r"\textit{Temperature} ($ ^{\circ} C $)", ylim_p=ylim_p,
            ylim_s=ylim_s, leg_labels=["Residual error", "Temperature"])

      p = np.polyfit(temp[j].astype(float), resid_o3[:, 0].astype(float), 1)
      r2 = stats.coeff_deter(temp[j].astype(float), resid_o3[:, 0].astype(float))
  
      plot_str = "$ e = %.3f * T + %.3f $" % (p[0], p[1])
      plot_str2 = "$ {R^2}_{eT} = %.4f $" % r2
      ax.text((epochs[-1] + 3*epochs[0])/4, (ylim_p[0] + 19*ylim_p[1])/20,
              plot_str, ha="center", va="bottom")
      ax.text((epochs[-1] + 3*epochs[0])/4, (ylim_p[0] + 19*ylim_p[1])/20,
              plot_str2, ha="center", va="top")
      
      ax.annotate("$ e = y_{pred} - y_{true} $", xy=(0.9, 0.9),
                  xycoords="axes fraction")
    

    no2_figs.append(fig_n)
    no2_fignames.append('no2-sens%d-res-temp-comp.svg' % (j+1))
    
    o3_figs.append(fig_o)
    o3_fignames.append('o3-sens%d-res-temp-comp.svg' % (j+1))
    
    # plot autocorrelation of residuals
    print "plotting autocorrelation of residuals"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plotting.set_plot_labels(ax, title="Autocorrelation of $ NO_2 $ residuals",
        xlabel="Lag", ylabel=r"\textit{Autocorrelation}")
    fig = plot_acf(pd.Series(resid_no2).values, ax=ax, lags=np.arange(0, 2000, 10))

    no2_figs.append(fig)
    no2_fignames.append("no2-sens%d-autocorr.svg" % (j+1))

    #print "plotting PACF"
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plotting.set_plot_labels(ax, title="PACF of $ NO_2 $ residuals",
    #    xlabel="Lag", ylabel=r"\textit{PACF}")
    #fig = plot_pacf(pd.Series(resid_no2).values, ax=ax, lags=np.arange(0, 2000, 10))

    #no2_figs.append(fig)
    #no2_fignames.append("no2-sens%d-pacf.svg" % (j+1))
  # -------------------------------------------------------------------------

    # time-series forecast residuals ARIMA
    #print "computing ARIMA model"
    #arima_model = ARIMA(resid_no2, order=(1440, 0, 1440))
    #model_fit = arima_model.fit(disp=5)

    #print "forecasting training set values"
    #predict_resid_no2 = model_fit.predict(start=None, end=None)
    #predict_resid_no2 = np.concatenate(([0], predict_resid_no2), axis=0)

    #plot_var = np.zeros([np.size(resid_no2), 2])
    #plot_var[:, 0] = resid_no2
    #plot_var[:, 1] = predict_resid_no2

    #fig = plt.figure()
    #ax = fig.add_subplot('111')
    #ax.plot(epochs, resid_no2, 'b')
    #ax.plot(epochs, predict_resid_no2, 'r')
    #fig.show()

  # ---------------------------- VISUALIZATION ------------------------------
    #print "plotting forecasted model"
    #fig, ax = plotting.ts_plot(epochs, plot_var,
    #      title = r'\textbf{Time-series forecasted} $ NO_2 $ \textbf{residuals: ARIMA model}',
    #      ylabel= r'\textit{Residual error}',
    #      leg_labels=['Actual error', 'Forecasted error'])

    #txt = get_corr_txt(plot_var[:, 1], plot_var[:, 0])
    #ax.annotate(txt, xy=(0.7, 0.75), xycoords='axes fraction')

    #no2_figs.append(fig)
    #no2_fignames.append('no2-sens%d-arima-forecast.svg' % (j+1))
  
    #fig = plotting.plot_violin(resid_no2,
    #       title="Violin-plot of residual errors from multifold regression",
    #       xlabel="Runs", ylabel="Residual error", scale=[-50, 50])

    #figs.append(fig)
    print "Sensor %d DONE" % (j+1)


  # plot comparison of predicted values
  no2_y_pred = np.array(no2_y_pred).T
  o3_y_pred = np.array(o3_y_pred).T

  # TODO: Make legend labels more generic
  if len(no2_x) > 1:
    fig, ax = plotting.ts_plot(epochs, no2_y_pred,
          title = r'\textbf{Comparison of } $ NO_2 $ \textbf{ predictions from SATVAM sensors}',
          ylabel= r'\textit{Concentration of } $ NO_2 $  (ppb)',
          leg_labels=[("Sensor %d" % x) for x in range(1, len(no2_x)+1)])
    
    txt = get_corr_txt(no2_y_pred[:, 0], no2_y_pred[:, 1])
    ax.annotate(txt, xy=(0.7, 0.75), xycoords='axes fraction')

    no2_figs.append(fig)
    no2_fignames.append('no2-predicted-comp.svg')

    # TODO: Make legend labels more generic
    fig, ax = plotting.ts_plot(epochs, o3_y_pred,
          title = r'\textbf{Comparison of } $ O_3 $ \textbf{ predictions from SATVAM sensors}',
          ylabel= r'\textit{Concentration of } $ O_3 $  (ppb)',
          leg_labels=[("Sensor %d" % x) for x in range(1, len(no2_x)+1)])
    
    txt = get_corr_txt(o3_y_pred[:, 0], o3_y_pred[:, 1])
    ax.annotate(txt, xy=(0.7, 0.75), xycoords='axes fraction')

    o3_figs.append(fig)
    o3_fignames.append('o3-predicted-comp.svg')

  # compare violins of each NO2 sensor
  coeffs_no2 = np.array(coeffs_no2)
  coeffs_ox = np.array(coeffs_ox)

  mean_no2_coeffs = []
  mean_ox_coeffs = []
  print coeffs_no2_names
  for i in xrange(coeffs_no2.shape[2]):
    fig, means, medians = plotting.plot_violin(coeffs_no2[:, :, i].T,
        title=r"\textbf{Coefficient of %s for } $ NO_2 $" % coeffs_no2_names[i],
        ylabel=r"\textit{Coefficient of %s}" % coeffs_no2_names[i],
        xlabel=r"\textbf{Sensors}")

    print means
    mean_no2_coeffs.append(means)
    #print medians

    no2_figs.append(fig)
    no2_fignames.append('no2-coeff%d-violin.svg' % (i + 1))

  mean_no2_coeffs = np.array(mean_no2_coeffs)

  for i in xrange(coeffs_ox.shape[2]):
    fig, means, medians = plotting.plot_violin(coeffs_ox[:, :, i].T,
        title=r"\textbf{Coefficient of %s for } $ O_3 $" % coeffs_ox_names[i],
        ylabel=r"\textit{Coefficient of %s}" % coeffs_ox_names[i],
        xlabel=r"\textbf{Sensors}")

    print means
    mean_ox_coeffs.append(means)
    #print medians

    o3_figs.append(fig)
    o3_fignames.append('o3-coeff%d-violin.svg' % (i + 1))

  mean_ox_coeffs = np.array(mean_ox_coeffs).T
  # -------------------------------------------------------------------------


  # plot time series of sensors predicted using all different coefficients
  for (j, no2) in enumerate(no2_x):
    
    leg_labels = [('Coefficient set %d' % i) for i in range(1,
          np.shape(mean_no2_coeffs)[1] + 1)]
    leg_labels.insert(0, 'Reference conc')

    no2 = np.concatenate((no2, np.ones([np.shape(no2)[0], 1])), axis=1)
    t_series = np.concatenate((np.reshape(no2_y, [np.size(no2_y), 1]),
        np.dot(no2, mean_no2_coeffs)), axis=1)

    fig, ax = plotting.ts_plot(epochs, t_series,
        title = r'\textbf{Comparison of predicted values for }'
              + r'$ NO_2 $ \textbf{ (Sensor %d)}' % (j + 1),
        ylabel = r'Concentration (ppb)',
        leg_labels=leg_labels)

    for i in range(1, np.size(t_series, axis=1)):
      text = get_corr_txt(t_series[:, i].astype(float),
        t_series[:, 0].astype(float), add_title='Set %d' % i)

      x = i / 5.0
      ax.annotate(text, xy = (x, 0.75), xycoords='axes fraction')

    no2_figs.append(fig)
    no2_fignames.append('no2-coeffs-predict-sens%d' % (j + 1))

  return no2_figs, no2_fignames, o3_figs, o3_fignames

# ----------------------------------------------------------------------------------
def pm_correlate(data, ref_pm1_incl=False, ref_pm10_incl=False):
  
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

  figs.append(fig)
  fignames.append('pm1-comp.svg')

  fig, ax = plotting.ts_plot(ts, pm10_vals,
    title = r'$ PM_{10} $ concentration',
    ylabel = r'\textit{Concentration ($ ug/m^3 $)}',
    leg_labels=leg_labels_pm10)

  for i in range(1, np.size(pm25_vals, axis=1)):
    text = get_corr_txt(pm25_vals[:, i].astype(float),
        pm25_vals[:, 0].astype(float), add_title='S%d' % i)

    x = i / 5.0
    ax.annotate(text, xy = (x, 0.75), xycoords='axes fraction')

  figs.append(fig)
  fignames.append('pm10-comp.svg')

  fig, ax = plotting.ts_plot(ts, pm25_vals,
    title = r'$ PM_{2.5} $ concentration',
    ylabel = r'\textit{Concentration ($ ug/m^3 $)}',
    leg_labels=leg_labels)

  for i in range(1, np.size(pm25_vals, axis=1)):
    text = get_corr_txt(pm25_vals[:, i].astype(float),
        pm25_vals[:, 0].astype(float), add_title='S%d' % i)

    x = i / 5.0
    ax.annotate(text, xy = (x, 0.75), xycoords='axes fraction')

  figs.append(fig)
  fignames.append('pm25-comp.svg')

  return figs, fignames
# ----------------------------------------------------------------------------------
