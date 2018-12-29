# -------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import matplotlib as mpl
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker

import time
import scipy.stats

import plotting
# ------------------------------------------------------------------------------
def parse_ts(ts):
  epoch = int(time.mktime(time.strptime(ts, '%d/%m/%y %H:%M')))
  return epoch
# ------------------------------------------------------------------------------
# import raw data
raw_data = pd.read_csv('../no2_delhi_deployment_withref.csv')
#raw_data = pd.read_csv("../satvam-comparison-10-dec-formatted.csv") #, usecols=[3,4,9], skiprows=3)
data = raw_data[raw_data.applymap(lambda x: 
              (x != "NoData" and x != "RS232" 
           and x != "NO_DATA" and x != "CALIB_S"
           and x != "CALIB_Z" and x != "FAULTY")).all(1)].dropna()

# for some reason the ref data is read as strings
data['refsensor(ppb)'] = data['refsensor(ppb)'].map(float)

# remove sub-zero ppb values
data = data[data.applymap(lambda x: x > 0).all(1)]

# remove possible outliers: Not corroborated with theory!
#tmp_data = pd.DataFrame(data.iloc[:, 0])
#data = data[tmp_data.applymap(lambda x: x < 200).all(1)]
# ------------------------------------------------------------------------------
training_set_ratio = 0.7
train_size = int(np.size(data, 0) * training_set_ratio)

print "Training set size: \t", train_size
print "Test set size: \t", (np.size(data, 0) - train_size)

no2_x = []

# parse time strings/
#epochs = data.values[:,0]
ts = data.values[:,0]
epochs = []
for current_ts in ts:
  epochs.append(parse_ts(current_ts))
epochs = np.array(epochs)

temps = data.values[:, 2] / 1000
#print temps

# store x and y values
no2_y = data.values[:,1]
for i in range(3, np.size(data.values, 1), 2):
  no2_x.append(data.values[:,i:(i + 2)])

we_zero_offsets = [227, 229, 222]
ae_zero_offsets = [228, 230, 221]
# ------------------------------------------------------------------------------
runs = 10
residuals = np.zeros([len(no2_y), runs])
coeffs = []

for (j, sensor) in enumerate(no2_x):
  print "\n----------- Sensor: " + str(j + 1) + "---------------"
  coeffs.append([])

  for i in xrange(runs):
  
    #print "\n---------- Run : " + str(i + 1) + "----------------"
    data_values = np.random.permutation(data.values)
  
    #print 2*j+2, 2*(j+2)
    no2_x_train = data_values[:train_size,(2*j+2):(2*j+4)]
    no2_y_train = data_values[:train_size,1]
  
    no2_x_test = data_values[train_size:,(2*j+2):(2*j+4)]
    no2_y_test = data_values[train_size:,1]

    # subtracting zero offsets
    no2_x_train[0] -= we_zero_offsets[j]
    no2_x_train[1] -= ae_zero_offsets[j]
  
    # train model
    reg = LinearRegression().fit(no2_x_train, no2_y_train)
  
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

  # plot residuals wrt time
  print "Done"
  ylim_p = [-150, 50]
  ylim_s = [10, 50]
  fig, ax = plotting.compare_ts_plot(epochs, residuals[:, 0], temps,
          title="Residual errors vs temperature (Sensor " + str(j + 1) + ")",
          xlabel="Time Stamp", ylabel="Residuals (ppb)",
          ylabel_s="Temperature (\deg C)", ylim_p=ylim_p,
          ylim_s=ylim_s)

  # compute r^2 between residual and temperature
  p = np.polyfit(temps.astype(float), residuals[:, 0].astype(float), 1)
  print (np.corrcoef(temps.astype(float), residuals[:, 0].astype(float))[0,1])
  r2 = (np.corrcoef(temps.astype(float), residuals[:, 0].astype(float))[0,1])

  plot_str = "rs = %.3f * T + %.3f" % (p[0], p[1])
  plot_str2 = "R^2 = %.4f" % r2
  ax.text((epochs[-1] + epochs[0])/2, (4*ylim_p[0] + ylim_p[1])/5, plot_str, ha="center", va="bottom")
  ax.text((epochs[-1] + epochs[0])/2, (4*ylim_p[0] + ylim_p[1])/5, plot_str2, ha="center", va="top")
  
  fig.show()
  
  fig = plt.figure()
  autocorrelation_plot(pd.Series(residuals[:,0]))

# -------------------------------------------------------------------------------
# plot last obtained surface
fig = plotting.plot_reg_plane(no2_x_train, no2_y_train,
                     [reg.intercept_, reg.coef_[0], reg.coef_[1]],
                     title='Outputs vs. ppb characteristic',
                     xlabel='sensor output WE (mV)',
                     ylabel='sensor output AE (mV)',
                     zlabel='NO2 concentration (ppb)')
fig.show()

fig = plotting.plot_reg_plane(no2_x_train[:, 0], no2_y_train,
                     [reg.intercept_, reg.coef_[0]],
                     title='Outputs vs. ppb characteristic',
                     xlabel='sensor output WE (mV)',
                     ylabel='NO2 concentration (ppb)')
fig.show()
#ax = fig.add_subplot(111, projection='3d')
#
## plot the training set
#ax.scatter(no2_x_train[:,0], no2_x_train[:,1], no2_y_train, color='r')
#
## plot the surface
#plot_plane(ax, [np.min(no2_x_train[:,0]), np.min(no2_x_train[:,1])],
#           [np.max(no2_x_train[:,0]), np.max(no2_x_train[:,1])],
#           [reg.intercept_, reg.coef_[0], reg.coef_[1]])
#
#ax.set_xlabel('sensor out WE (mV)')
#ax.set_ylabel('sensor out AE (mV)')
#ax.set_zlabel('NO2 concentration (ppb)')
#ax.set_title('Outputs vs. ppb characteristic')
#
#fig.show()
## -------------------------------------------------------------------------------
##plot test set
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
## plot the training set
#ax.scatter(no2_x_test[:,0], no2_x_test[:,1], no2_y_test, color='r')
#
## plot the surface
#plot_plane(ax, [np.min(no2_x_test[:,0]), np.min(no2_x_test[:,1])],
#           [np.max(no2_x_test[:,0]), np.max(no2_x_test[:,1])],
#           [reg.intercept_, reg.coef_[0], reg.coef_[1]])
#
#ax.set_xlabel('sensor out WE (mV)')
#ax.set_ylabel('sensor out AE (mV)')
#ax.set_zlabel('NO2 concentration (ppb)')
#ax.set_title('Outputs vs. ppb on test set')
#
#fig.show()
# -------------------------------------------------------------------------------
#plotting.plot_violin(residuals, title="Violin-plot of residual errors from multifold regression",
#            xlabel="Runs", ylabel="Residual error", scale=[-50, 50])
coeffs = np.array(coeffs)

fig = plotting.plot_violin(coeffs[:, :, 0].T, title="Coefficient of sensor output 1 from multifold regression",
            ylabel="Coefficient of sensor-output 1 (ppb/mV)", xlabel="Sensors")
fig.show()

fig = plotting.plot_violin(coeffs[:, :, 1].T, title="Coefficient of sensor output 2 from multifold regression",
            ylabel="Coefficient of sensor-output 2 (ppb/mV)", xlabel="Sensors")
fig.show()

fig = plotting.plot_violin(coeffs[:, :, 2].T, title="Constant term from multifold regression",
            ylabel="Constant term (ppb)", xlabel="Sensors")
fig.show()
# -------------------------------------------------------------------------------
