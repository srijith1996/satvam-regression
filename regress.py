# ------------------------------------------------------------------------------
import pandas as pd
from pandas.plotting import autocorrelation_plot
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib as mpl

import time

import plotting
# ------------------------------------------------------------------------------
def regress_df(data, runs = 1000):

  # list of all figures plotted
  figs = []
  
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
  ox_y = data.values[:,2]
  for i in xrange(np.size(data.values, 1)):
    if (5*i + col_skip) >= np.size(data.values, 1):
      break

    temp.append(data.values[:,col_temp(i)])
    no2_x.append(data.values[:,col_no2(i)])
    ox_x.append(data.values[:,col_ox(i)])


  # TODO
  #we_zero_offsets = [227, 229, 222]
  #ae_zero_offsets = [228, 230, 221]

  residuals = np.zeros([len(no2_y), runs])
  coeffs = []
  
  for j in xrange(len(no2_x)):
    print "Sensor: " + str(j + 1)
    coeffs.append([])
  
    for i in xrange(runs):
    
      data_values = np.random.permutation(data.values)
    
      no2_x_train = data_values[:train_size, col_no2(j)]
      no2_y_train = data_values[:train_size, 1]
    
      no2_x_test = data_values[train_size:, col_no2(j)]
      no2_y_test = data_values[train_size:, 1]
  
      # TODO: subtracting zero offsets
      #no2_x_train[0] -= we_zero_offsets[j]
      #no2_x_train[1] -= ae_zero_offsets[j]
    
      # train model
      reg = LinearRegression().fit(no2_x_train, no2_y_train)
    
      # TODO: Publish to report
      #print "Regression coefficients: \t", reg.coef_
      #print "Regression intercept: \t", reg.intercept_
      #print "coeff of AE: \t", -(reg.coef_[1] /reg.coef_[0])
      #print "sensitivity: \t", 1/reg.coef_[0]
    
      # test model
      no2_out_test = reg.predict(no2_x_test)
      dev_test = np.sum(np.square(no2_out_test - no2_y_test))
      dev_test = np.sqrt(dev_test)/np.size(no2_y_test, 0)
      
      abs_dev_test = np.sum(np.abs(no2_out_test - 
              no2_y_test)) / np.size(no2_y_test, 0)
      
      # training model
      no2_out_train = reg.predict(no2_x_train)
      dev_train = np.sum(np.square(no2_out_train - no2_y_train))
      dev_train = np.sqrt(dev_train)/np.size(no2_y_train, 0)
      
      abs_dev_train = np.sum(np.abs(no2_out_train -
          no2_y_train) / no2_out_train)/np.size(no2_y_train, 0)
      
      #print "Deviation measure L2 training set: \t", dev_train
      #print "Deviation measure L2 test set: \t", dev_test
      #print "Deviation measure L1 training set: \t", abs_dev_train
      #print "Deviation measure L1 test set: \t", abs_dev_test
      
      #print "R^2 training set: \t", reg.score(no2_x_train, no2_y_train)
      #print "R^2 test set: \t", reg.score(no2_x_test, no2_y_test)
    
      residuals[:, i] = reg.predict(no2_x[j]) - no2_y
      coeffs[j].append([reg.coef_[0], reg.coef_[1], reg.intercept_])
  
    print "DONE"

    # plot residuals wrt time
    ylim_p = [-150, 50]
    ylim_s = [0, 45]
    fig, ax = plotting.compare_ts_plot(epochs, residuals[:, 0], temp[j],
          title="Residual errors vs temperature (Sensor " + str(j + 1) + ")",
          xlabel="Time Stamp", ylabel="Residuals (ppb)",
          ylabel_s="Temperature (\deg C)", ylim_p=ylim_p,
          ylim_s=ylim_s, leg_labels=["Residual error", "Temperature"])
  
    # compute r^2 between residual and temperature
    p = np.polyfit(temp[j].astype(float), residuals[:, 0].astype(float), 1)
    print (np.corrcoef(temp[j].astype(float),
        residuals[:, 0].astype(float))[0,1])
    r2 = (np.corrcoef(temp[j].astype(float),
        residuals[:, 0].astype(float))[0,1])
  
    plot_str = "rs = %.3f * T + %.3f" % (p[0], p[1])
    plot_str2 = "R^2 = %.4f" % r2
    ax.text((epochs[-1] + epochs[0])/2, (4*ylim_p[0] + ylim_p[1])/5,
            plot_str, ha="center", va="bottom")
    ax.text((epochs[-1] + epochs[0])/2, (4*ylim_p[0] + ylim_p[1])/5,
            plot_str2, ha="center", va="top")
    
    figs.append(fig)
    fig.show()
    
    # plot autocorrelation of residuals
    fig = plt.figure()
    ax = autocorrelation_plot(pd.Series(residuals[:,0]))
    plotting.set_plot_labels(ax, title="Autocorrelation of residuals",
        xlabel="Lag", ylabel="Autocorrelation")

    figs.append(fig)
    fig.show()
  
    # plot regression surface
    fig = plotting.plot_reg_plane(no2_x_train, no2_y_train,
                     [reg.intercept_, reg.coef_[0], reg.coef_[1]],
                     title='Outputs vs. ppb on training set',
                     xlabel='sensor output WE (mV)',
                     ylabel='sensor output AE (mV)',
                     zlabel='NO2 concentration (ppb)')
    figs.append(fig)
    fig.show()

    fig = plotting.plot_reg_plane(no2_x_test, no2_y_test,
                     [reg.intercept_, reg.coef_[0], reg.coef_[1]],
                     title='Outputs vs. ppb on test set',
                     xlabel='sensor output WE (mV)',
                     ylabel='sensor output AE (mV)',
                     zlabel='NO2 concentration (ppb)')
    figs.append(fig)
    fig.show()

    #fig = plotting.plot_violin(residuals,
    #       title="Violin-plot of residual errors from multifold regression",
    #       xlabel="Runs", ylabel="Residual error", scale=[-50, 50])

    #figs.append(fig)
    #fig.show()

  # compare violins of each sensor
  coeffs = np.array(coeffs)

  fig = plotting.plot_violin(coeffs[:, :, 0].T,
      title="Coefficient of sensor output 1 from multifold regression",
      ylabel="Coefficient of sensor-output 1 (ppb/mV)", xlabel="Sensors")

  figs.append(fig)
  fig.show()

  fig = plotting.plot_violin(coeffs[:, :, 1].T,
      title="Coefficient of sensor output 2 from multifold regression",
      ylabel="Coefficient of sensor-output 2 (ppb/mV)", xlabel="Sensors")
  
  figs.append(fig)
  fig.show()

  fig = plotting.plot_violin(coeffs[:, :, 2].T,
      title="Constant term from multifold regression",
      ylabel="Constant term (ppb)", xlabel="Sensors")
  
  figs.append(fig)
  fig.show()

  return figs
# ----------------------------------------------------------------------------------
