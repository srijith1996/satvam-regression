# ------------------------------------------------------------------------------
import pandas as pd
from pandas.plotting import autocorrelation_plot
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

import time

import plotting
import stats
# ------------------------------------------------------------------------------
def get_corr_txt(y_true, y_pred, add_title=''):

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
def regress_df(data, runs = 1000):

  # list of all figures plotted
  no2_figs = []
  o3_figs = []

  no2_fignames = []
  o3_fignames = []
  
  # remove sub-zero ppb values
  data = data[data.applymap(lambda x: x > 0).all(1)]
  
  # remove possible outliers
  #tmp_data = pd.DataFrame(data.iloc[:, 0])
  #data = data[tmp_data.applymap(lambda x: x < 200).all(1)]

  training_set_ratio = 0.7
  train_size = int(np.size(data, 0) * training_set_ratio)

  print "Training set size: \t", train_size
  print "Test set size: \t", (np.size(data, 0) - train_size)

  no2_x = []
  temp = []
  ox_x = []

  # column locations for no2, ox and temperature data of the ith sensor
  col_skip = 3
  col_no2 = (lambda i: range((col_skip + 5*i + 1),(col_skip + 5*i + 3)))
  col_ox = (lambda i: range((col_skip + 5*i + 3),(col_skip + 5*i + 5)))
  col_temp = (lambda i: (col_skip + 5*i))

  epochs = data.values[:,0]

  # store x and y values
  no2_y = data.values[:,1]
  o3_y = data.values[:,2]
  for i in xrange(np.size(data.values, 1)):
    if (5*i + col_skip) >= np.size(data.values, 1):
      break

    temp.append(data.values[:,col_temp(i)])
    no2_x.append(data.values[:,col_no2(i)])
    ox_x.append(data.values[:,col_ox(i)])

  # convert o3 to ox for regression
  ox_y = o3_y + no2_y

  # visualize time-series of ref data
  fig, ax = plotting.ts_plot(epochs, no2_y,
         title=r'$ NO_2 $\textbf{ readings from Reference monitor}',
         ylabel=r'$ NO_2 $\textit{ concentration (ppb)}',
         leg_labels=[r'$ NO_2 $ conc (ppb)'])

  no2_figs.append(fig)
  no2_fignames.append('no2-ref.svg')

  fig, ax = plotting.ts_plot(epochs, o3_y,
         title=r'$ O_3 $ \textbf{readings from Reference monitor}',
         ylabel=r'$ O_3 $\textit{concentration (ppb)}',
         leg_labels=['$ O_3 $ conc (ppb)'])

  o3_figs.append(fig)
  o3_fignames.append('o3-ref.svg')

  # visualize time-series of AlphaSense sensors
  no2_op1_vals = np.zeros([np.shape(no2_x[0])[0], len(no2_x)])
  no2_op2_vals = np.zeros([np.shape(no2_x[0])[0], len(no2_x)])
  ox_op1_vals = np.zeros([np.shape(ox_x[0])[0], len(ox_x)])
  ox_op2_vals = np.zeros([np.shape(ox_x[0])[0], len(ox_x)])

  for (i, sens_no2) in enumerate(no2_x):
    no2_op1_vals[:, i] = sens_no2[:, 0]
    no2_op2_vals[:, i] = sens_no2[:, 1]

  # TODO: Change labels based on sensor name 
  fig, ax = plotting.ts_plot(epochs, no2_op1_vals,
         title=r'$ NO_2 $ \textbf{Output1 from AlphaSense sensors}',
         ylabel=r'$ NO_2 $ \textit{op1 (mV)}', ylim=[190, 300],
         leg_labels=[('Sensor %d' % x) for x in range(1, len(no2_x)+1)])

  no2_figs.append(fig)
  no2_fignames.append('no2-op1.svg')

  # visualize time-series of AlphaSense sensors
  fig, ax = plotting.ts_plot(epochs, no2_op2_vals,
         title=r'$ NO_2 $ \textbf{Output2 from AlphaSense sensors}',
         ylabel=r'$ NO_2 $ \textit{op2 (mV)}', ylim=[190, 300],
         leg_labels=[('Sensor %d' % x) for x in range(1, len(no2_x)+1)])

  no2_figs.append(fig)
  no2_fignames.append('no2-op2.svg')

  for (i, sens_ox) in enumerate(ox_x):
    ox_op1_vals[:, i] = sens_ox[:, 0]
    ox_op2_vals[:, i] = sens_ox[:, 1]

  # TODO: Change labels based on sensor name 
  fig, ax = plotting.ts_plot(epochs, ox_op1_vals,
         title=r'$ OX (NO_2 + O_3) $ \textbf{Output 1 from AlphaSense sensors}',
         ylabel=r'$ OX $ \textit{op1 (mV)}', ylim=[190, 300],
         leg_labels=[('Sensor %d' % x) for x in range(1, len(ox_x)+1)])

  o3_figs.append(fig)
  o3_fignames.append('o3-op1.svg')

  # visualize time-series of AlphaSense sensors
  fig, ax = plotting.ts_plot(epochs, ox_op2_vals,
         title=r'$ OX (NO_2 + O_3) $ \textbf{Output 2 from AlphaSense sensors}',
         ylabel=r'$ OX $ \textit{op2 (mV)}', ylim=[190, 300],
         leg_labels=[('Sensor %d' % x) for x in range(1, len(ox_x)+1)])

  o3_figs.append(fig)
  o3_fignames.append('o3-op2.svg')


  # process and regress data: multifold

  # TODO
  #we_zero_offsets = [227, 229, 222]
  #ae_zero_offsets = [228, 230, 221]

  resid_no2 = np.zeros([len(no2_y), runs])
  resid_o3 = np.zeros([len(ox_y), runs])
  coeffs_no2 = []
  coeffs_ox = []

  predict_no2 = []
  predict_o3 = []
  
  for j in xrange(len(no2_x)):
    print "Sensor: " + str(j + 1)
    coeffs_no2.append([])
    coeffs_ox.append([])
  
    for i in xrange(runs):
    
      data_values = np.random.permutation(data.values)
    
      no2_x_train = data_values[:train_size, col_no2(j)]
      no2_y_train = data_values[:train_size, 1]
      no2_x_test = data_values[train_size:, col_no2(j)]
      no2_y_test = data_values[train_size:, 1]

      ox_x_train = data_values[:train_size, col_ox(j)]
      ox_y_train = data_values[:train_size, 2] + no2_y_train
      ox_x_test = data_values[train_size:, col_ox(j)]
      ox_y_test = data_values[train_size:, 2] + no2_y_test
  
      # TODO: subtracting zero offsets
      #no2_x_train[0] -= we_zero_offsets[j]
      #no2_x_train[1] -= ae_zero_offsets[j]
    
      # train model
      reg_no2 = LinearRegression().fit(no2_x_train, no2_y_train)
      reg_ox = LinearRegression().fit(ox_x_train, ox_y_train)
    
      # TODO: Publish to report
      #print "Regression coefficients: \t", reg_no2.coef_
      #print "Regression intercept: \t", reg_no2.intercept_
      #print "coeff of AE: \t", -(reg_no2.coef_[1] /reg_no2.coef_[0])
      #print "sensitivity: \t", 1/reg_no2.coef_[0]
    
      # test model
      no2_out_test = reg_no2.predict(no2_x_test)
      dev_test = np.sum(np.square(no2_out_test - no2_y_test))
      dev_test = np.sqrt(dev_test)/np.size(no2_y_test, 0)

      abs_dev_test = np.sum(np.abs(no2_out_test - 
              no2_y_test)) / np.size(no2_y_test, 0)
      
      # training model
      no2_out_train = reg_no2.predict(no2_x_train)
      dev_train = np.sum(np.square(no2_out_train - no2_y_train))
      dev_train = np.sqrt(dev_train)/np.size(no2_y_train, 0)
      
      abs_dev_train = np.sum(np.abs(no2_out_train -
          no2_y_train) / no2_out_train)/np.size(no2_y_train, 0)
      
      #print "Deviation measure L2 training set: \t", dev_train
      #print "Deviation measure L2 test set: \t", dev_test
      #print "Deviation measure L1 training set: \t", abs_dev_train
      #print "Deviation measure L1 test set: \t", abs_dev_test
      
      #print "R^2 training set: \t", reg_no2.score(no2_x_train, no2_y_train)
      #print "R^2 test set: \t", reg_no2.score(no2_x_test, no2_y_test)
    
      predict_no2 = reg_no2.predict(no2_x[j])
      predict_o3 = reg_ox.predict(ox_x[j]) - predict_no2

      resid_no2[:, i] = predict_no2 - no2_y
      resid_o3[:, i] = predict_o3 - o3_y

      coeffs_no2[j].append([reg_no2.coef_[0], reg_no2.coef_[1], reg_no2.intercept_])
      coeffs_ox[j].append([reg_ox.coef_[0], reg_ox.coef_[1], reg_ox.intercept_])

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
    no2_fignames.append('no2-predict-true-comp.svg')

    print "plotting actual and predicted values: O3"
    t_series = np.array([o3_y, predict_o3]).T

    fig, ax = plotting.ts_plot(epochs, t_series,
        title = r'\textbf{True and Predicted concentrations of } $ O_3 $',
        ylabel = r'Concentration (ppb)',
        leg_labels=['Reference conc.', 'Predicted conc.'])

    text = get_corr_txt(t_series[:, 0], t_series[:, 1])
    ax.annotate(text, xy = (0.7, 0.75), xycoords='axes fraction')

    o3_figs.append(fig)
    o3_fignames.append('o3-predict-true-comp.svg')

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
    fig, ax = plotting.compare_ts_plot(epochs, resid_no2[:, 0], temp[j],
          title=r"\textbf{Prediction errors (} $ NO_2 $ \textbf{) vs temperature (Sensor "
               + str(j + 1) + ")}",
          ylabel=r"\textit{Residuals (ppb)}",
          ylabel_s=r"\textit{Temperature} ($ ^{\circ} C $)", ylim_p=ylim_p,
          ylim_s=ylim_s, leg_labels=["Residual error", "Temperature"])

    # compute r^2 between residual and temperature
    print "computing residual vs temperature correlation"
    p = np.polyfit(temp[j].astype(float), resid_no2[:, 0].astype(float), 1)
    r2 = stats.coeff_deter(temp[j].astype(float), resid_no2[:, 0].astype(float))
  
    plot_str = "$ e = %.3f * T + %.3f $" % (p[0], p[1])
    plot_str2 = "$ {R^2}_{eT} = %.4f $" % r2
    ax.text((epochs[-1] + 3*epochs[0])/4, (ylim_p[0] + 19*ylim_p[1])/20,
            plot_str, ha="center", va="bottom")
    ax.text((epochs[-1] + 3*epochs[0])/4, (ylim_p[0] + 19*ylim_p[1])/20,
            plot_str2, ha="center", va="top")

    ax.annotate("$ e = y_{pred} - y_{true} $", xy=(0.8, 0.9),
                xycoords="axes fraction")
    
    no2_figs.append(fig)
    no2_fignames.append('no2-res-temp-comp.svg')
    
    # plot residuals wrt time for O3
    print "plotting residual characteristics"
    ylim_p = [-150, 50]
    ylim_s = [0, 45]
    fig, ax = plotting.compare_ts_plot(epochs, resid_o3[:, 0], temp[j],
          title=r"\textbf{Prediction errors (} $ O_3 $ \textbf{) vs temperature (Sensor "
               + str(j + 1) + ")}",
          ylabel=r"\textit{Residuals (ppb)}",
          ylabel_s=r"\textit{Temperature} ($ ^{\circ} C $)", ylim_p=ylim_p,
          ylim_s=ylim_s, leg_labels=["Residual error", "Temperature"])

    # compute r^2 between residual and temperature
    print "computing residual vs temperature correlation"
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
    
    o3_figs.append(fig)
    o3_fignames.append('o3-res-temp-comp.svg')
    
    # plot autocorrelation of residuals
    #print "plotting autocorrelation of residuals"
    #fig = plt.figure()
    #ax = autocorrelation_plot(pd.Series(resid_no2[:,0]))
    #plotting.set_plot_labels(ax, title="Autocorrelation of resid_no2",
    #    xlabel="Lag", ylabel=r"\textit{Autocorrelation}")
    #ax.set_xlim(0, 250)

    #figs.append(fig)

    # time-series forecast residuals ARIMA
    print "computing ARIMA model"
    arima_model = ARIMA(resid_no2[:, 0], order=(5, 1, 1))
    model_fit = arima_model.fit(disp=5)

    print "forecasting training set values"
    predict_resid_no2 = model_fit.predict(start=None, end=None)
    predict_resid_no2 = np.concatenate(([0], predict_resid_no2), axis=0)

    plot_var = np.zeros([np.size(resid_no2[:, 0]), 2])
    plot_var[:, 0] = resid_no2[:,0]
    plot_var[:, 1] = predict_resid_no2

    #fig = plt.figure()
    #ax = fig.add_subplot('111')
    #ax.plot(epochs, resid_no2[:, 0], 'b')
    #ax.plot(epochs, predict_resid_no2, 'r')
    #fig.show()

    print "plotting forecasted model"
    fig, ax = plotting.ts_plot(epochs, plot_var,
          title = r'\textbf{Time-series forecasted} $ NO_2 $ \textbf{residuals: ARIMA model}',
          ylabel= r'\textit{Residual error}',
          leg_labels=['Actual error', 'Forecasted error'])

    txt = get_corr_txt(plot_var[:, 1], plot_var[:, 0])
    ax.annotate(txt, xy=(0.7, 0.75), xycoords='axes fraction')

    no2_figs.append(fig)
    no2_fignames.append('no2-arima-forecast.svg')
  
    #fig = plotting.plot_violin(resid_no2,
    #       title="Violin-plot of residual errors from multifold regression",
    #       xlabel="Runs", ylabel="Residual error", scale=[-50, 50])

    #figs.append(fig)
    print "Sensor %d DONE" % (j+1)

  # compare violins of each NO2 sensor
  coeffs_no2 = np.array(coeffs_no2)

  fig = plotting.plot_violin(coeffs_no2[:, :, 0].T,
      title=r"\textbf{Coefficient of sensor op1 for } $ NO_2 $",
      ylabel=r"\textit{Coefficient of sensor op1 (ppb/mV)}",
      xlabel=r"\textbf{Sensors}")

  no2_figs.append(fig)
  no2_fignames.append('no2-coeff1-violin.svg')

  fig = plotting.plot_violin(coeffs_no2[:, :, 1].T,
      title=r"\textbf{Coefficient of sensor op2 for } $ NO_2 $",
      ylabel=r"\textit{Coefficient of sensor op2 (ppb/mV)}",
      xlabel=r"\textbf{Sensors}")
  
  no2_figs.append(fig)
  no2_fignames.append('no2-coeff2-violin.svg')

  fig = plotting.plot_violin(coeffs_no2[:, :, 2].T,
      title=r"\textbf{Constant term for } $ NO_2 $",
      ylabel=r"\textit{Constant term (ppb)}",
      xlabel=r"\textbf{Sensors}")
  
  no2_figs.append(fig)
  no2_fignames.append('no2-const-violin.svg')

  # compare violins of each O3 sensor
  coeffs_ox = np.array(coeffs_ox)

  fig = plotting.plot_violin(coeffs_ox[:, :, 0].T,
      title=r"\textbf{Coefficient of sensor op1 for } $ OX $",
      ylabel=r"\textit{Coefficient of sensor op1 (ppb/mV)}",
      xlabel=r"\textbf{Sensors}")

  o3_figs.append(fig)
  o3_fignames.append('o3-coeff1-violin.svg')

  fig = plotting.plot_violin(coeffs_ox[:, :, 1].T,
      title=r"\textbf{Coefficient of sensor op2 for } $ OX $",
      ylabel=r"\textit{Coefficient of sensor op2 (ppb/mV)}",
      xlabel=r"\textbf{Sensors}")
  
  o3_figs.append(fig)
  o3_fignames.append('o3-coeff2-violin.svg')

  fig = plotting.plot_violin(coeffs_ox[:, :, 2].T,
      title=r"\textbf{Constant term for } $ OX $",
      ylabel=r"\textit{Constant term (ppb)}",
      xlabel=r"\textbf{Sensors}")
  
  o3_figs.append(fig)
  o3_fignames.append('o3-const-violin.svg')

  return no2_figs, no2_fignames, o3_figs, o3_fignames
# ----------------------------------------------------------------------------------
def pm_correlate(data):
  
  # list of all figures plotted
  figs = []
  
  # remove sub-zero ppb values
  data = data[data.applymap(lambda x: x > 0).all(1)]
  
  ts = data.values[:,0]
  pm1_vals = []
  pm25_vals = []
  pm10_vals = []

  leg_labels = []

  pm25_vals.append(data.values[:, 1])
  for i in range(2, np.size(data.values, 1), 3):
    pm1_vals.append(data.values[:, i].tolist())
    pm25_vals.append(data.values[:, i+1].tolist())
    pm10_vals.append(data.values[:, i+2].tolist())

    leg_labels.append('Sensor %d' % int(((i-2)/3) + 1))

  pm1_vals = np.array(pm1_vals).T
  pm25_vals = np.array(pm25_vals).T
  pm10_vals = np.array(pm10_vals).T

  fig, ax = plotting.ts_plot(ts, pm1_vals,
    title = r'$ PM_{1.0} $ concentration',
    ylabel = r'\textit{Concentration ($ ug/m^3 $)}',
    leg_labels=leg_labels)


  figs.append(fig)

  fig, ax = plotting.ts_plot(ts, pm10_vals,
    title = r'$ PM_{10} $ concentration',
    ylabel = r'\textit{Concentration ($ ug/m^3 $)}',
    leg_labels=leg_labels)


  figs.append(fig)

  leg_labels.insert(0, 'EBAM')

  fig, ax = plotting.ts_plot(ts, pm25_vals,
    title = r'$ PM_{2.5} $ concentration',
    ylabel = r'\textit{Concentration ($ ug/m^3 $)}',
    leg_labels=leg_labels)

  text = r'\textbf{Correlation coefficients}'

  for i in range(1, np.size(pm25_vals, axis=1)):
    text = get_corr_txt(pm25_vals[:, i].astype(float),
        pm25_vals[:, 0].astype(float), add_title='S%d' % i)

    x = i / 5.0
    ax.annotate(text, xy = (x, 0.75), xycoords='axes fraction')

  figs.append(fig)

  return figs
# ----------------------------------------------------------------------------------
