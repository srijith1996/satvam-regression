# ------------------------------------------------------------------------------
import stats
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------
# configuration
NTn = 0.6
KTn = 1.0
K_Tn = 1.0
K__Tn = 0
NTo = 1.7
KTo = 1.0
K_To = 1.0
K__To = 0

NO2_WE_0T = [227, 227]
NO2_AE_0T = [225, 225]
NO2_WE_0E = [227, 227]
NO2_AE_0E = [228, 228]
SENSITIVITY_NO2 = [0.270, 0.261]
# ------------------------------------------------------------------------------
O3_WE_0T = [244, 244]
O3_AE_0T = [249, 249]
O3_WE_0E = [239, 239]
O3_AE_0E = [244, 244]
SENSITIVITY_O3 = [0.357, 0.321]
# ------------------------------------------------------------------------------
def formula1(we_raw, ae_raw, we_0e, ae_0e, nt, sens):
  we_raw -= we_0e
  ae_raw -= ae_0e
  we_raw -= nt * ae_raw
  plt.plot(we_raw/sens);
  plt.show();
  return we_raw/sens
# ------------------------------------------------------------------------------
def formula2(we_raw, ae_raw, we_0e, ae_0e, we_0t, ae_0t, kt, sens):
  we_0 = we_0t - we_0e
  ae_0 = ae_0t - ae_0e

  we_raw -= we_0e
  ae_raw -= ae_0e
  we_raw -= kt * ae_raw * (we_0 / ae_0)
  plt.plot(we_raw/sens);
  plt.show();
  return we_raw/sens
# ------------------------------------------------------------------------------
def formula3(we_raw, ae_raw, we_0e, ae_0e, we_0t, ae_0t, kt, sens):
  we_0 = we_0t - we_0e
  ae_0 = ae_0t - ae_0e

  we_raw -= we_0e
  ae_raw -= ae_0e
  we_raw -= kt * ae_raw - (we_0 - ae_0)
  plt.plot(we_raw/sens);
  plt.show();
  return we_raw/sens
# ------------------------------------------------------------------------------
def formula4(we_raw, ae_raw, we_0e, ae_0e, we_0t, ae_0t, kt, sens):
  we_0 = we_0t - we_0e

  we_raw -= we_0e
  ae_raw -= ae_0e
  we_raw -= kt * ae_raw - we_0
  return we_raw/sens
# ------------------------------------------------------------------------------
def alphasense_compute(dataFrame, t_incl=False, h_incl=False):

  # lambdas for columns
  col_skip = 3
  if t_incl and h_incl:
    col_temp = (lambda i: (col_skip + 6*i))
    col_hum = (lambda i: (col_skip + 6*i + 1))
    col_no2 = (lambda i: range((col_skip + 6*i + 2),(col_skip + 6*i + 4)))
    col_ox = (lambda i: range((col_skip + 6*i + 4),(col_skip + 6*i + 6)))
  elif h_incl:
    col_hum = (lambda i: (col_skip + 5*i))
    col_no2 = (lambda i: range((col_skip + 5*i + 1),(col_skip + 5*i + 3)))
    col_ox = (lambda i: range((col_skip + 5*i + 3),(col_skip + 5*i + 5)))
  elif t_incl:
    col_temp = (lambda i: (col_skip + 5*i))
    col_no2 = (lambda i: range((col_skip + 5*i + 1),(col_skip + 5*i + 3)))
    col_ox = (lambda i: range((col_skip + 5*i + 3),(col_skip + 5*i + 5)))
  else:
    col_no2 = (lambda i: range((col_skip + 4*i),(col_skip + 4*i + 2)))
    col_ox = (lambda i: range((col_skip + 4*i + 2),(col_skip + 4*i + 4)))

  dataFrame = dataFrame.values
  err_no2 = np.zeros([len(NO2_WE_0T), 4, 3])
  err_o3 = np.zeros([len(NO2_WE_0T), 4, 3])

  # iterate over sensors
  for i in xrange(np.size(dataFrame, 1)):
    if col_ox(i)[-1] >= np.size(dataFrame, 1):
      break

    nx = dataFrame[:, col_no2(i)]
    ox = dataFrame[:, col_ox(i)]

    # formula 1
    pred_no2 = formula1(nx[:, 0], nx[:, 1], NO2_WE_0E[i], NO2_AE_0E[i],
                        NTn, SENSITIVITY_NO2[i])
    err_no2[i, 0, 0] = stats.mae(dataFrame[:, 1], pred_no2)
    err_no2[i, 0, 1] = stats.rmse(dataFrame[:, 1], pred_no2)
    err_no2[i, 0, 2] = stats.mape(dataFrame[:, 1], pred_no2)

    pred = formula1(ox[:, 0], ox[:, 1], O3_WE_0E[i], O3_AE_0E[i],
                    NTo, SENSITIVITY_O3[i])
    err_o3[i, 0, 0] = stats.mae(dataFrame[:, 2], pred - pred_no2)
    err_o3[i, 0, 1] = stats.rmse(dataFrame[:, 2], pred - pred_no2)
    err_o3[i, 0, 2] = stats.mape(dataFrame[:, 2], pred - pred_no2)

    # formula 2
    pred_no2 = formula2(nx[:, 0], nx[:, 1], NO2_WE_0E[i], NO2_AE_0E[i],
                    NO2_WE_0T[i], NO2_AE_0T[i], KTn, SENSITIVITY_NO2[i])
    err_no2[i, 1, 0] = stats.mae(dataFrame[:, 1], pred)
    err_no2[i, 1, 1] = stats.rmse(dataFrame[:, 1], pred)
    err_no2[i, 1, 2] = stats.mape(dataFrame[:, 1], pred)

    pred = formula2(ox[:, 0], ox[:, 1], O3_WE_0E[i], O3_AE_0E[i],
                    O3_WE_0T[i], O3_AE_0T[i], KTo, SENSITIVITY_O3[i])
    err_o3[i, 1, 0] = stats.mae(dataFrame[:, 2], pred - pred_no2)
    err_o3[i, 1, 1] = stats.rmse(dataFrame[:, 2], pred - pred_no2)
    err_o3[i, 1, 2] = stats.mape(dataFrame[:, 2], pred - pred_no2)

    # formula 3
    pred_no2 = formula3(nx[:, 0], nx[:, 1], NO2_WE_0E[i], NO2_AE_0E[i],
                    NO2_WE_0T[i], NO2_AE_0T[i], K_Tn, SENSITIVITY_NO2[i])
    err_no2[i, 2, 0] = stats.mae(dataFrame[:, 1], pred_no2)
    err_no2[i, 2, 1] = stats.rmse(dataFrame[:, 1], pred_no2)
    err_no2[i, 2, 2] = stats.mape(dataFrame[:, 1], pred_no2)

    pred = formula3(ox[:, 0], ox[:, 1], O3_WE_0E[i], O3_AE_0E[i],
                    O3_WE_0T[i], O3_AE_0T[i], K_To, SENSITIVITY_O3[i])
    err_o3[i, 2, 0] = stats.mae(dataFrame[:, 2], pred - pred_no2)
    err_o3[i, 2, 1] = stats.rmse(dataFrame[:, 2], pred - pred_no2)
    err_o3[i, 2, 2] = stats.mape(dataFrame[:, 2], pred - pred_no2)

    # formula 4
    pred_no2 = formula4(nx[:, 0], nx[:, 1], NO2_WE_0E[i], NO2_AE_0E[i],
                    NO2_WE_0T[i], NO2_AE_0T[i], K__Tn, SENSITIVITY_NO2[i])
    err_no2[i, 3, 0] = stats.mae(dataFrame[:, 1], pred_no2)
    err_no2[i, 3, 1] = stats.rmse(dataFrame[:, 1], pred_no2)
    err_no2[i, 3, 2] = stats.mape(dataFrame[:, 1], pred_no2)

    pred = formula4(ox[:, 0], ox[:, 1], O3_WE_0E[i], O3_AE_0E[i],
                    O3_WE_0T[i], O3_AE_0T[i], K__To, SENSITIVITY_O3[i])
    err_o3[i, 3, 0] = stats.mae(dataFrame[:, 2], pred - pred_no2)
    err_o3[i, 3, 1] = stats.rmse(dataFrame[:, 2], pred - pred_no2)
    err_o3[i, 3, 2] = stats.mape(dataFrame[:, 2], pred - pred_no2)

  print err_no2
  print err_o3
# ------------------------------------------------------------------------------
