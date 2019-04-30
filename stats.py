# ------------------------------------------------------------------------------
import numpy as np
from math import log10, floor
# ------------------------------------------------------------------------------
def round_sig(x, sig=2):

  return np.round(x, sig - int(floor(log10(abs(x)))) - 1)
# ------------------------------------------------------------------------------
def rmse(y_true, y_pred):
  
  rmse = np.sqrt(np.sum(np.square(y_pred
      - y_true))/np.size(y_true))

  return rmse
# ------------------------------------------------------------------------------
def mape(y_true, y_pred):
  
  mape = np.sum(np.abs((y_pred - y_true)/y_true))
  mape = mape * 100 / np.size(y_true)

  return mape
# ------------------------------------------------------------------------------
def mae(y_true, y_pred):
  
  mae = np.sum(np.abs(y_true - y_pred))
  mae = mae / np.size(y_true)

  return mae
# ------------------------------------------------------------------------------
def pearson(y_true, y_pred):
  
  N = np.size(y_true)

  num = np.sum(y_true * y_pred) * N - np.sum(y_true) * np.sum(y_pred)
  den = (N * np.sum(np.square(y_true))) - np.square(np.sum(y_true))
  den = den * ((N * np.sum(np.square(y_pred))) - np.square(np.sum(y_pred)))
  den = np.sqrt(den)

  return (num/den)
# ------------------------------------------------------------------------------
def coeff_deter(y_true, y_pred):

  mu = np.mean(y_true)
  ss_res = np.sum(np.square(y_true - y_pred))
  ss_tot = np.sum(np.square(y_true - mu))

  r2 = 1 - (ss_res / ss_tot)

  return r2
# ------------------------------------------------------------------------------
