# -------------------------------------------------------------------------------
'''
  This script runs on raw files from SATVAM deployement and does regression on the data.
  Functionality sequence:
    - Read and convert timestamps on all input files
    - Timestamp align all files with respect to one another
    - Run regression and publish inferences

  Usage:
    python autoreg.py <ref_data_file> [<ebam_file>]
            <space_separated_list_of_sensor_files> <output-files-prefix>

    use [<ebam_file>] only when DEPLOY_SITE is set to 'MRIU'

'''
# -------------------------------------------------------------------------------
import gc
import pandas as pd
import numpy as np
import sys, os
import datetime as dt
from datetime import timedelta
import time
import regress
#import pdfpublish
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import alphasense
import plotting
# -------------------------------------------------------------------------------
# Change this field to 'MPCB' or 'MRIU' based on deployment site
PARSE_TEMP_REF = True
DEPLOY_SITE = 'MRIU'
DEPLOYMENT = 2
CONF_AVG_WINDOW_SIZE_MIN = 1
CONF_CLEAN = 3
CONF_RUNS = 10
CONF_TEMP_HUM_FILE = 'ref'            # either 'ref' or 'satvam'

DEPLOY_SITE = DEPLOY_SITE.upper()

# configurables
STEP_SIZE_TS = 60                     # desired granularity of data input

# SATVAM graphana export fields
TIME_FIELD_HDR = 'Time'               # Header name of Time field 
NO2OP1_FIELD_HDR = 'no2op1'           # Header name of no2 output1 field 
NO2OP2_FIELD_HDR = 'no2op2'           # Header name of no2 output2 field 
OXOP1_FIELD_HDR = 'o3op1'             # Header name of o3 output1 field 
OXOP2_FIELD_HDR = 'o3op2'             # Header name of o3 output2 field 
TEMP_FIELD_HDR = 'temp'               # Header name of temperature field
HUM_FIELD_HDR = 'humidity'            # Header name of humidity field

PM1_FIELD_HDR = 'pm1conc'
PM25_FIELD_HDR = 'pm25conc'
PM10_FIELD_HDR = 'pm10conc'

SENS_TS_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'  # Input timestamp format
DES_FORMAT = '%Y/%m/%d %H:%M:%S'      # Output format

#reference monitor data fields
if DEPLOY_SITE == 'MRIU':

  # input arg files
  REF_FILE = sys.argv[1]
  EBAM_FILE = None
  SENSOR_FILE_LIST = None

  if DEPLOYMENT > 1:
    EBAM_FILE = sys.argv[2]
    if CONF_TEMP_HUM_FILE == 'ref':
      REF_TH_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
      REF_TH_DELIM = r'\s+'
      REF_TH_COL_DATE = 0
      REF_TH_COL_TIME = 1
      REF_TH_COL_TEMP = 6
      REF_TH_COL_HUM = 3
      TEMP_HUM_FILE = sys.argv[3]
      SENSOR_FILE_LIST = sys.argv[4:-1]
    else:
      SENSOR_FILE_LIST = sys.argv[3:-1]
  else:
    SENSOR_FILE_LIST = sys.argv[2:-1]

  R_SKIP_ROWS = [7]
  R_SKIP_ROWS_END = 11
  R_HEADER_ROW = 6
  R_DATE_FIELD_HDR = 'Date'
  R_TIME_FIELD_HDR = 'Time'
  R_OX_FIELD_HDR = 'OZONE'
  R_NO2_FIELD_HDR = 'NO2'

  REF_TS_FORMAT = '%m/%d/%y %H:%M'
  REF_DATE_FORMAT = '%m/%d/%y'
  EBAM_TS_FORMAT = '%d-%m-%Y %H:%M'
  #EBAM_TS_FORMAT = '%d/%m/%y %H:%M'

  # EBAM data fields
  EBAM_HEADER_ROW = 0
  EBAM_TS_FIELD_HDR = 'Time'
  EBAM_PM25_FIELD_HDR = 'ConcRT(mg/m3)'


elif DEPLOY_SITE == 'MPCB':

  # input arg files
  REF_FILE = sys.argv[1]
  #EBAM_FILE = sys.argv[2]

  R_SKIP_ROWS = []
  R_SKIP_ROWS_END = 8
  R_HEADER_ROW = 0
  R_TIME_FIELD_HDR = 'Date Time'
  R_OX_FIELD_HDR = 'O3 [ug/m3]'
  R_NO2_FIELD_HDR = 'NO2 [ug/m3]'
  R_PM25_FIELD_HDR = 'PM25 [ug/m3]'
  R_PM10_FIELD_HDR = 'PM10 [ug/m3]'

  REF_TS_FORMAT = '%d-%m-%Y %H:%M'

  SENSOR_FILE_LIST = sys.argv[2:-1]


OUT_FILE_PREFIX = sys.argv[-1]
DIR_PREFIX = sys.argv[-1] + '/'

# prepare output directory
try:
  os.stat(DIR_PREFIX)
except:
  os.makedirs(DIR_PREFIX)

NUM_SENSORS = len(SENSOR_FILE_LIST)
# -------------------------------------------------------------------------------
def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    pass

  return False 
# -------------------------------------------------------------------------------
# ---------------- PRE-PROCESS SATVAM SENSOR DATA -------------------------------
# -------------------------------------------------------------------------------
print "Processing SATVAM data (%d sensors)........" % NUM_SENSORS

# list of dataframes for each mote
sens_dfs = []
min_time = max_time = 0

print "Interpreting time stamps......"
for (i, src_file) in enumerate(SENSOR_FILE_LIST):
  #target_file = src_file[:src_file.index('.csv')] + "-fmt.csv"

  # read file 
  sens_dfs.append(pd.read_csv(src_file, sep=';', header=1))

  timestamps = sens_dfs[i][TIME_FIELD_HDR].values

  # convert timezone to IST from Z
  for (j, timestamp) in enumerate(timestamps):
    timestamps[j] = (dt.datetime.strptime(timestamp,
        SENS_TS_FORMAT) + timedelta(hours=5, minutes=30)).strftime('%s')
    timestamps[j] = int(timestamps[j])
    
    # change resolution to minutes
    timestamps[j] -= timestamps[j] % 60

  sens_dfs[i][TIME_FIELD_HDR] = pd.DataFrame(data=timestamps)
  #print sens_dfs[i]

  if DEPLOYMENT == 1:
    sens_dfs[i] = sens_dfs[i].drop([PM1_FIELD_HDR, PM25_FIELD_HDR,
          PM10_FIELD_HDR, TEMP_FIELD_HDR, HUM_FIELD_HDR], axis=1)

  sens_dfs[i] = sens_dfs[i][sens_dfs[i].applymap(lambda x: 
            (x != "NoData" and x != "NO_DATA"
         and x != "undefined")).all(1)].dropna()

  print "Sensor %d has %d points" % (i, len(sens_dfs[i].index))

  # first iteration
  if i == 0:
    min_time = timestamps[0]
    max_time = timestamps[-1]

  # find the minimum and maximum epochs
  if min_time > timestamps[0]:
    min_time = timestamps[0]
  if max_time < timestamps[-1]:
    min_time = timestamps[-1]

#min_time -= min_time % 60
#max_time -= (max_time % 60) + 60

print "DONE"
# -------------------------------------------------------------------------------
# ----------------- PRE-PROCESS REFERENCE MONITOR DATA --------------------------
# -------------------------------------------------------------------------------
print "Processing Reference monitor data........"

ref_remove = ["NoData", "NO_DATA", "RS232", "CALIB_S", "CALIB_Z",
              "FAULTY", "Samp<", "MAINT", "Zero", "Calib", "Span"]

ref_df = pd.read_excel(REF_FILE, header=R_HEADER_ROW, skiprows=R_SKIP_ROWS)
ref_df = ref_df.drop(np.arange(len(ref_df) - R_SKIP_ROWS_END, len(ref_df)))

if DEPLOYMENT == 1 and DEPLOY_SITE == 'MRIU':
  ref_df = ref_df.drop(["SO2", "CO"], axis=1)
ref_df = ref_df[ref_df.applymap(lambda x:
            (x not in ref_remove)).all(1)].dropna()

# clean up time values that are 24:00
for index, row in ref_df.iterrows():
  if row[R_TIME_FIELD_HDR] == '24:00' or row[R_TIME_FIELD_HDR][-5:] == '24:00':
    if DEPLOY_SITE == 'MRIU':
      row[R_TIME_FIELD_HDR] = '00:00'
      row[R_DATE_FIELD_HDR] = dt.datetime.strptime(
          row[R_DATE_FIELD_HDR], REF_DATE_FORMAT) + timedelta(days=1)
      row[R_DATE_FIELD_HDR] = dt.datetime.strftime(row[R_DATE_FIELD_HDR],
          REF_DATE_FORMAT)

    elif DEPLOY_SITE == 'MPCB':
      row[R_TIME_FIELD_HDR] = row[R_TIME_FIELD_HDR][:-5] + '00:00'
      row[R_TIME_FIELD_HDR] = dt.datetime.strptime(
          row[R_TIME_FIELD_HDR], REF_TS_FORMAT) + timedelta(days=1)
      row[R_TIME_FIELD_HDR] = dt.datetime.strftime(row[R_TIME_FIELD_HDR],
          REF_TS_FORMAT)

print "Interpreting time stamps...."
times = ref_df[R_TIME_FIELD_HDR].values

if DEPLOY_SITE == 'MRIU':
  dates = ref_df[R_DATE_FIELD_HDR].values

for i in xrange(len(times)):

  if DEPLOY_SITE == 'MRIU':
    times[i] = dates[i] + ' ' + times[i]

  times[i] = dt.datetime.strptime(times[i], REF_TS_FORMAT).strftime('%s')
  times[i] = int(times[i])

  # change resolution to minutes
  times[i] -= times[i] % 60

print "Reference monitor has %d points" % len(ref_df.index)

if DEPLOY_SITE == 'MRIU':
  ref_df = ref_df.drop(columns=[R_DATE_FIELD_HDR])

min_time = min([times[0], min_time])
max_time = max([times[-1], max_time])

#print min_time, max_time

print "DONE"
#print ref_df
# -------------------------------------------------------------------------------
# ----------------- PRE-PROCESS EBAM DATA ---------------------------------------
# -------------------------------------------------------------------------------
if DEPLOY_SITE == 'MRIU' and DEPLOYMENT != 1:
  print "Processing EBAM data........"
  
  ebam_df = pd.read_csv(EBAM_FILE, header=EBAM_HEADER_ROW)
  ebam_df = ebam_df[[EBAM_TS_FIELD_HDR, EBAM_PM25_FIELD_HDR]].dropna()

  # These strings do not exist in the file,
  # This statement gets rid of the "string not float" error
  ebam_df = ebam_df[ebam_df.applymap(lambda x:
              (x != "NoData" and x != "NO_DATA"
           and x != "RS232" and x != "CALIB_S"
           and x != "CALIB_Z" and x != "FAULTY"
           and x != "Samp<")).all(1)].dropna()
  
  print "Interpreting time stamps...."
  times = ebam_df[EBAM_TS_FIELD_HDR].values

  for i in xrange(len(times)):
    times[i] = dt.datetime.strptime(times[i],
        EBAM_TS_FORMAT).strftime('%s')
    times[i] = int(times[i])
      
    # change resolution to minutes
    times[i] -= times[i] % 60
  
  print "EBAM has %d points" % len(ebam_df.index)
  
  min_time = min([times[0], min_time])
  max_time = max([times[-1], max_time])
  
  #print min_time, max_time
  
  print "DONE"

# -------------------------------------------------------------------------------
# ----------------- PRE-PROCESS Reference T/RH file -----------------------------
# -------------------------------------------------------------------------------
if DEPLOYMENT == 2 and DEPLOY_SITE == 'MRIU' and CONF_TEMP_HUM_FILE == 'ref':
  print "Processing Vaisala reference data........"
  
  refth_df = pd.read_csv(TEMP_HUM_FILE, sep=REF_TH_DELIM, header=None)
  refth_df = refth_df[[REF_TH_COL_DATE, REF_TH_COL_TIME,
                       REF_TH_COL_TEMP, REF_TH_COL_HUM]].dropna()

  refth_df = refth_df[refth_df.applymap(lambda x: (x != "'C")).all(1)].dropna()

  print "Interpreting time stamps...."
  dates = refth_df[REF_TH_COL_DATE].values
  times = refth_df[REF_TH_COL_TIME].values

  for i in xrange(len(times)):

    times[i] = dates[i] + ' ' + times[i]
    times[i] = dt.datetime.strptime(times[i], REF_TH_TIME_FORMAT).strftime('%s')
    times[i] = int(times[i])
      
    # change resolution to minutes
    times[i] -= times[i] % 60
  
  print "Vaisala Reference has %d points" % len(refth_df.index)
  refth_df = refth_df.drop(columns=[REF_TH_COL_DATE])
  
  min_time = min([times[0], min_time])
  max_time = max([times[-1], max_time])
  
  print "DONE"
# -------------------------------------------------------------------------------
# generate the time vector
time_vec = np.arange(min_time, max_time+60, 60)
index_ts = lambda x: (x-min_time)/60
#print time.strftime(DES_FORMAT, time.gmtime(time_vec[-1]))

# copy data for each sensor 
no2_op1 = np.empty([len(time_vec), NUM_SENSORS])
no2_op2 = np.empty([len(time_vec), NUM_SENSORS])
ox_op1 = np.empty([len(time_vec), NUM_SENSORS])
ox_op2 = np.empty([len(time_vec), NUM_SENSORS])

no2_op1[:] = no2_op2[:] = ox_op1[:] = ox_op2[:] = np.nan

if DEPLOYMENT != 1:
  temp = np.empty([len(time_vec), NUM_SENSORS])
  hum = np.empty([len(time_vec), NUM_SENSORS])
  pm1 = np.empty([len(time_vec), NUM_SENSORS])
  pm25 = np.empty([len(time_vec), NUM_SENSORS])
  pm10 = np.empty([len(time_vec), NUM_SENSORS])
  sens_t = np.empty([len(time_vec), NUM_SENSORS])
  sens_h = np.empty([len(time_vec), NUM_SENSORS])

  pm1[:] = pm25[:] = pm10[:] = temp[:] = np.nan

#  a = np.empty([len(time_vec), NUM_SENSORS + 2])
#  b = np.empty([len(time_vec), NUM_SENSORS + 2])
#  a[:] = b[:] = np.nan

#a[:, 0] = time_vec
#b[:, 0] = time_vec

print "Time-stamp aligning SATVAM sensor values...."
for i in xrange(NUM_SENSORS):
  
  print "Sensor " + str(i + 1)
  # collect values
  sens_ts = sens_dfs[i][TIME_FIELD_HDR].values
  sens_no2op1 = sens_dfs[i][NO2OP1_FIELD_HDR].values
  sens_no2op2 = sens_dfs[i][NO2OP2_FIELD_HDR].values
  sens_oxop1 = sens_dfs[i][OXOP1_FIELD_HDR].values
  sens_oxop2 = sens_dfs[i][OXOP2_FIELD_HDR].values

  if DEPLOYMENT != 1:
    #if CONF_TEMP_HUM_FILE != 'ref':
    sens_temp = sens_dfs[i][TEMP_FIELD_HDR].values
    sens_hum = sens_dfs[i][HUM_FIELD_HDR].values
    sens_pm1 = sens_dfs[i][PM1_FIELD_HDR].values
    sens_pm25 = sens_dfs[i][PM25_FIELD_HDR].values
    sens_pm10 = sens_dfs[i][PM10_FIELD_HDR].values

  # align to time_vec
  for j in xrange(len(sens_ts)):
    #ts_index = time_vec.tolist().index(sens_ts[j])
    ts_index = index_ts(sens_ts[j])
    no2_op1[ts_index, i] = sens_no2op1[j]
    no2_op2[ts_index, i] = sens_no2op2[j]
    ox_op1[ts_index, i] = sens_oxop1[j]
    ox_op2[ts_index, i] = sens_oxop2[j]

    if DEPLOYMENT != 1:
      sens_t[ts_index, i] = sens_temp[j]
      sens_h[ts_index, i] = sens_hum[j]
      if DEPLOY_SITE != 'MRIU' or CONF_TEMP_HUM_FILE != 'ref':
        temp[ts_index, i] = sens_temp[j]
        hum[ts_index, i] = sens_hum[j]
#      a[ts_index, i+1] = sens_temp[j]
#      b[ts_index, i+1] = sens_hum[j]
      pm1[ts_index, i] = sens_pm1[j]
      pm25[ts_index, i] = sens_pm25[j]
      pm10[ts_index, i] = sens_pm10[j]

#print no2_op1
print "DONE"
# -------------------------------------------------------------------------------
if DEPLOYMENT == 2 and DEPLOY_SITE == 'MRIU' and CONF_TEMP_HUM_FILE == 'ref':
  print "Time-stamp aligning Reference T/RH values...."

  refth_ts = refth_df[REF_TH_COL_TIME].values
  refth_temp = refth_df[REF_TH_COL_TEMP].values
  refth_hum = refth_df[REF_TH_COL_HUM].values

  for j in xrange(len(refth_ts)):
    ts_index = index_ts(refth_ts[j])
#    a[ts_index, NUM_SENSORS + 1] = refth_temp[j]
#    b[ts_index, NUM_SENSORS + 1] = refth_hum[j]
    for i in xrange(NUM_SENSORS):
      temp[ts_index, i] = refth_temp[j]
      hum[ts_index, i] = refth_hum[j]

  print "DONE"

# plot temperature and humidity comparison
#a = a[(a > 0).all(axis=1)]
#a = a[~np.isnan(a).any(axis=1)]
#
#b = b[(b > 0).all(axis=1)]
#b = b[~np.isnan(b).any(axis=1)]
#labs = [("Sensor %d" % s) for s in range(1, NUM_SENSORS + 1)]
#labs.append("Reference")
#fig1, ax1  = plotting.ts_plot(a[:, 0], a[:, 1:],
#                       title=r'\textbf{Comparison of temperature readings}',
#                       ylabel=r'\textit{Temperature }($^{\circ} C $)',
#                       leg_labels=labs, ids=[5, 7, 9])

#for i in range(2, np.size(a, axis=1)):
#  text = regress.get_corr_txt(a[:, i].astype(float),
#      a[:, 1].astype(float), add_title='S%d' % (i-1))

#  x = i / 5.0
#  ax1.annotate(text, xy = (x, 0.75), xycoords='axes fraction')

#fig2, ax2  = plotting.ts_plot(b[:, 0], b[:, 1:],
#                       title=r'\textbf{Comparison of RH readings}',
#                       ylabel=r'\textit{RH }\%',
#                       leg_labels=labs, ids=[5, 7, 9])

#for i in range(2, np.size(b, axis=1)):
#  text = regress.get_corr_txt(b[:, i].astype(float),
#      b[:, 1].astype(float), add_title='S%d' % (i-1))

#  x = i / 5.0
#  ax2.annotate(text, xy = (x, 0.75), xycoords='axes fraction')

#plt.show()
# -------------------------------------------------------------------------------
print "Time-stamp aligning ref monitor data...."

ref_ts = ref_df[R_TIME_FIELD_HDR].values
ref_no2 = np.empty([len(time_vec), ])
ref_o3 = np.empty([len(time_vec), ])

ref_no2[:] = ref_o3[:] = np.nan

ref_pm10 = np.empty([len(time_vec), ])
ref_pm25 = np.empty([len(time_vec), ])

ref_pm25[:] = ref_pm10[:] = np.nan

for j in xrange(len(ref_ts)):
  ts_index = index_ts(ref_ts[j])
  ref_no2[ts_index] = ref_df[R_NO2_FIELD_HDR].values[j]
  ref_o3[ts_index] = ref_df[R_OX_FIELD_HDR].values[j]

  if DEPLOY_SITE == 'MPCB':
    ref_pm10[ts_index] = ref_df[R_PM10_FIELD_HDR].values[j]
    ref_pm25[ts_index] = ref_df[R_PM25_FIELD_HDR].values[j]

print "DONE"
# -------------------------------------------------------------------------------
if DEPLOY_SITE == 'MRIU' and DEPLOYMENT != 1:
  print "Time-stamp aligning EBAM data..."
  
  # ebam currently measures only pm2.5
  ebam_ts = ebam_df[EBAM_TS_FIELD_HDR].values
  
  for j in xrange(len(ebam_ts)):
    ts_index = index_ts(ebam_ts[j])
    
    # convert to ug/m3
    ref_pm25[ts_index] = ebam_df[EBAM_PM25_FIELD_HDR].values[j] * 1000
  
  print "DONE"
# -------------------------------------------------------------------------------
aggregate_list = []

aggregate_list.append(time_vec)
aggregate_list.append(ref_no2)
aggregate_list.append(ref_o3)

for i in xrange(NUM_SENSORS):
  if DEPLOYMENT != 1:
    aggregate_list.append(temp[:, i].tolist())
    aggregate_list.append(hum[:, i].tolist())
  aggregate_list.append(no2_op1[:, i].tolist())
  aggregate_list.append(no2_op2[:, i].tolist())
  aggregate_list.append(ox_op1[:, i].tolist())
  aggregate_list.append(ox_op2[:, i].tolist())

aggregate_list = np.array(aggregate_list)
if DEPLOYMENT != 1:
  aggregate_list_2 = np.zeros(aggregate_list.shape)
  aggregate_list_2[:] = aggregate_list
  for i in xrange(NUM_SENSORS):
    aggregate_list_2[3 + i * 6, :] = sens_t[:, i]
    aggregate_list_2[4 + i * 6, :] = sens_h[:, i]
# -------------------------------------------------------------------------------
target_df = pd.DataFrame(aggregate_list).transpose()
target_df = regress.preclean_df(target_df, cols=[1, 2, 5, 6, 7, 8,
                                                 11, 12, 13, 14])
target_df = regress.window_avg(target_df, CONF_AVG_WINDOW_SIZE_MIN)
target_df = target_df.dropna()
dataset_size = len(target_df.index)
print "Data set size (after dropna()): " + str(dataset_size)
# -------------------------------------------------------------------------------
print "Calling regression algorithm on obtained DataFrame"

if DEPLOYMENT != 1:
  target_df2 = pd.DataFrame(aggregate_list_2).transpose()
  target_df2 = target_df2.dropna()

target_dfs = [target_df, target_df2]

# free some memory
#del ref_no2, ref_o3, temp, hum, no2_op1, no2_op2, ox_op1, ox_op2
#del ref_df, refth_df, sens_dfs, ebam_df, aggregate_list
#gc.collect()

t_present = True
h_present = True
if DEPLOYMENT == 1:
  t_present = False
  h_present = False

#alphasense.alphasense_compute(target_df, t_incl=True, h_incl=True)

no2_figs, no2_names, o3_figs, o3_names = regress.regress_df(target_df,
        temps_present=t_present, incl_op1=True, incl_op2=True, 
        incl_temps=False, incl_op1t=False, incl_op2t=False,
        hum_present=h_present, incl_hum=False, incl_op1h=False, incl_op2h=False,
        incl_op12=False, incl_cross_terms=True, clean=CONF_CLEAN,
        runs=CONF_RUNS, loc_label=DEPLOY_SITE,
        ret_errs=False)

## -------------------------------------------------------------------------------
## error comparison for models with reference T/RH and sensor-boxes T/RH
## [op1, op2, t, op1t, op2t, h, op1h, op2h, op1op2, cross_terms]
#avg_duration = [60]
#vars_vector = np.array([[True, True, False, False, False,
#                         False, False, False, False, False],
#                        [True, True,  True, False, False,
#                         True, False, False, False, False],
#                        [True, True,  False, False, False,
#                         False, False, False, False,  True],
#                        [True, True,  True, False, False,
#                         True, False, False, False,  True]])
#var_names = np.array(['op1', 'op2', 't', 'op1t', 'op2t', 'h',
#                      'op1h', 'op2h', 'op1op2', 'cross_terms'])
#
#mapes_n = []
#mapes_o = []
#xlabs = []
#for avg in avg_duration:
#  for (i, df) in enumerate(target_dfs):
#    coeffs_nt, coeffs_ot, maes_no2_t, maes_o3_t, rmses_no2_t, rmses_o3_t, \
#    mapes_no2_t, mapes_o3_t, r2_no2_t, r2_o3_t, r_no2_t, r_o3_t \
#      = regress.regress_df(df,
#              temps_present=t_present, hum_present=h_present, clean=CONF_CLEAN,
#              runs=CONF_RUNS, loc_label=DEPLOY_SITE,
#              incl_op1=vars_vector[1, 0], incl_op2=vars_vector[1, 1],
#              incl_temps=vars_vector[1, 2], incl_op1t=vars_vector[1, 3],
#              incl_op2t=vars_vector[1, 4], incl_hum=vars_vector[1, 5],
#              incl_op1h=vars_vector[1, 6], incl_op2h=vars_vector[1, 7],
#              incl_op12=vars_vector[1, 8], incl_cross_terms=vars_vector[1, 9],
#              ret_errs=True)
#
#    mapes_n.append(mapes_no2_t.tolist())
#    mapes_o.append(mapes_o3_t.tolist())
#
#    if i == 0:
#      xlabs.append("Ref T:%d" % avg)
#    else:
#      xlabs.append("Sat T:%d" % avg)
#
#mapes_n = np.array(mapes_n)
#mapes_o = np.array(mapes_o)
#
#for i in xrange(NUM_SENSORS):
#  for j in xrange(NUM_SENSORS + 1):
#    test_str = str(j + 1)
#    if j == NUM_SENSORS:
#      test_str = "ALL"
#
#    fig, ax = plotting.plot_violin(mapes_n[:, i, :, j].T,
#          title = r"$ NO_2 $\textbf{ MAPE: Training sensor %s, Testing %d}" % \
#          (test_str, i + 1), ylabel = r"\textit{MAPE} (\%)",
#          x_tick_labels = xlabs)
#
#    txt = r"\textbf{Multifold runs}: %d" % CONF_RUNS + '\n'
#    txt = txt + r"\textbf{Dataset size (train + test)}: %d" % dataset_size + '\n'
#    txt = txt + r"\textbf{T} - \textit{Time duration for averaging}"
#
#    ax.annotate(txt, xy = (0.7, 0.8), xycoords='axes fraction')
#    fig.savefig(DIR_PREFIX + 'no2-train%stest%d.pdf' %
#            (test_str, i+1), format='pdf')
#
#    fig, ax = plotting.plot_violin(mapes_o[:, i, :, j].T,
#          title = r"$ OX $\textbf{ MAPE: Training sensor %s, Testing %d}" % \
#          (test_str, i + 1), ylabel = r"\textit{MAPE} (\%)",
#          x_tick_labels = xlabs)
#
#    ax.annotate(txt, xy = (0.7, 0.8), xycoords='axes fraction')
#    fig.savefig(DIR_PREFIX + 'o3-train%stest%d.pdf' %
#            (test_str, i+1), format='pdf')
#
#
## -------------------------------------------------------------------------------
## error comparison for different average duration and linear models
#coeffs_n = []
#coeffs_o = []
#maes_n = []
#maes_o = []
#rmses_n = []
#rmses_o = []
#mapes_n = []
#mapes_o = []
#r2_n = []
#r2_o = []
#r_n = []
#r_o = []
#means_n = []
#means_o = []
#stds_n = []
#stds_o = []
#xlabs = []
#for i in xrange(len(avg_duration)):
#  df = regress.window_avg(target_df, avg_duration[i])
#  df = target_df.dropna()
#  for j in xrange(len(vars_vector)):
#    print ("Avg duration: %d, Vars enabled: " % avg_duration[i])\
#            + str(var_names[vars_vector[j]])
#
#    coeffs_nt, coeffs_ot, maes_no2_t, maes_o3_t, rmses_no2_t, rmses_o3_t,\
#    mapes_no2_t, mapes_o3_t, r2_no2_t, r2_o3_t, r_no2_t, r_o3_t,\
#    mean_no2_t, mean_o3_t, std_no2_t, std_o3_t\
#          = regress.regress_df(df,
#              temps_present=t_present, hum_present=h_present, clean=CONF_CLEAN,
#              runs=CONF_RUNS, loc_label=DEPLOY_SITE,
#              incl_op1=vars_vector[j, 0], incl_op2=vars_vector[j, 1],
#              incl_temps=vars_vector[j, 2], incl_op1t=vars_vector[j, 3],
#              incl_op2t=vars_vector[j, 4], incl_hum=vars_vector[j, 5],
#              incl_op1h=vars_vector[j, 6], incl_op2h=vars_vector[j, 7],
#              incl_op12=vars_vector[j, 8], incl_cross_terms=vars_vector[j, 9],
#              ret_errs=True)
#
#    if coeffs_nt is not None:
#      repeats = coeffs_nt.shape[1] - 1
#      coeffs_nt = np.mean(coeffs_nt, axis=0)
#      coeffs_nt = np.repeat(coeffs_nt, repeats, axis=0)
#      coeffs_n.append(coeffs_nt.T)
#    else:
#      coeffs_n = None
#
#    if coeffs_ot is not None:
#      repeats = coeffs_ot.shape[1] - 1
#      coeffs_ot = np.mean(coeffs_ot, axis=0)
#      coeffs_ot = np.repeat(coeffs_ot, repeats, axis=0)
#      coeffs_o.append(coeffs_ot.T)
#    else:
#      coeffs_o = None
#
#    maes_n.append(maes_no2_t.tolist())
#    maes_o.append(maes_o3_t.tolist())
#    rmses_n.append(rmses_no2_t.tolist())
#    rmses_o.append(rmses_o3_t.tolist())
#    mapes_n.append(mapes_no2_t.tolist())
#    mapes_o.append(mapes_o3_t.tolist())
#    r2_n.append(r2_no2_t.tolist())
#    r2_o.append(r2_o3_t.tolist())
#    r_n.append(r_no2_t.tolist())
#    r_o.append(r_o3_t.tolist())
#    means_n.append(mean_no2_t.tolist())
#    means_o.append(mean_o3_t.tolist())
#    stds_n.append(std_no2_t.tolist())
#    stds_o.append(std_o3_t.tolist())
#
#    xlabs.append("T:%d C:%d" % (avg_duration[i],
#                                len(var_names[vars_vector[j]])))
#
#maes_n = np.array(maes_n)
#maes_o = np.array(maes_o)
#rmses_n = np.array(rmses_n)
#rmses_o = np.array(rmses_o)
#mapes_n = np.array(mapes_n)
#mapes_o = np.array(mapes_o)
#r2_n = np.array(r2_n)
#r2_o = np.array(r2_o)
#r_n = np.array(r_n)
#r_o = np.array(r_o)
#means_n = np.array(means_n)
#means_o = np.array(means_o)
#stds_n = np.array(stds_n)
#stds_o = np.array(stds_o)
#
#print means_n.shape, means_o.shape
#print stds_n.shape, stds_o.shape
## Save obtained coefficients in presentation table format
#for i in xrange(np.shape(r_n)[0]):
#  if coeffs_n is not None:
#    vec = coeffs_n[i]
#    tmp = np.mean(maes_n[i, :, :, :], axis=1).flatten('F')
#    tmp = np.reshape(tmp, [1, np.size(tmp)])
#    vec = np.concatenate((vec, tmp), axis=0)
#  else:
#    tmp = np.mean(maes_n[i, :, :, :], axis=1).flatten('F')
#    tmp = np.reshape(tmp, [1, np.size(tmp)])
#    vec = tmp
#
#  tmp = np.mean(rmses_n[i, :, :, :], axis=1).flatten('F')
#  tmp = np.reshape(tmp, [1, np.size(tmp)])
#  vec = np.concatenate((vec, tmp), axis=0)
#
#  tmp = np.mean(mapes_n[i, :, :, :], axis=1).flatten('F')
#  tmp = np.reshape(tmp, [1, np.size(tmp)])
#  vec = np.concatenate((vec, tmp), axis=0)
#
#  tmp = np.mean(r2_n[i, :, :, :], axis=1).flatten('F')
#  tmp = np.reshape(tmp, [1, np.size(tmp)])
#  vec = np.concatenate((vec, tmp), axis=0)
#
#  tmp = np.mean(r_n[i, :, :, :], axis=1).flatten('F')
#  tmp = np.reshape(tmp, [1, np.size(tmp)])
#  vec = np.concatenate((vec, tmp), axis=0)
#
#  tmp = means_n[i, :, :].flatten('F')
#  tmp = np.reshape(tmp, [1, np.size(tmp)])
#  vec = np.concatenate((vec, tmp), axis=0)
#
#  tmp = stds_n[i, :, :].flatten('F')
#  tmp = np.reshape(tmp, [1, np.size(tmp)])
#  vec = np.concatenate((vec, tmp), axis=0)
#
#  np.savetxt(DIR_PREFIX + "no2-iter%d" % i, vec, fmt='%0.4g', delimiter=',')
#
#for i in xrange(np.shape(r_o)[0]):
#  if coeffs_o is not None:
#    vec = coeffs_o[i]
#    tmp = np.mean(maes_o[i, :, :, :], axis=1).flatten('F')
#    tmp = np.reshape(tmp, [1, np.size(tmp)])
#    vec = np.concatenate((vec, tmp), axis=0)
#  else:
#    tmp = np.mean(maes_o[i, :, :, :], axis=1).flatten('F')
#    tmp = np.reshape(tmp, [1, np.size(tmp)])
#    vec = tmp
#
#  tmp = np.mean(rmses_o[i, :, :, :], axis=1).flatten('F')
#  tmp = np.reshape(tmp, [1, np.size(tmp)])
#  vec = np.concatenate((vec, tmp), axis=0)
#
#  tmp = np.mean(mapes_o[i, :, :, :], axis=1).flatten('F')
#  tmp = np.reshape(tmp, [1, np.size(tmp)])
#  vec = np.concatenate((vec, tmp), axis=0)
#
#  tmp = np.mean(r2_o[i, :, :, :], axis=1).flatten('F')
#  tmp = np.reshape(tmp, [1, np.size(tmp)])
#  vec = np.concatenate((vec, tmp), axis=0)
#
#  tmp = np.mean(r_o[i, :, :, :], axis=1).flatten('F')
#  tmp = np.reshape(tmp, [1, np.size(tmp)])
#  vec = np.concatenate((vec, tmp), axis=0)
#
#  tmp = means_o[i, :, :].flatten('F')
#  tmp = np.reshape(tmp, [1, np.size(tmp)])
#  vec = np.concatenate((vec, tmp), axis=0)
#
#  tmp = stds_o[i, :, :].flatten('F')
#  tmp = np.reshape(tmp, [1, np.size(tmp)])
#  vec = np.concatenate((vec, tmp), axis=0)
#
#  np.savetxt(DIR_PREFIX + "ox-iter%d" % i, vec, fmt='%0.4g', delimiter=',')
#
## -------------------------------------------------------------------------------
## plot MAPE errors for training different models and different avg durations
#training_sets = (NUM_SENSORS + 1) if (NUM_SENSORS > 1) else NUM_SENSORS
#
#for i in xrange(NUM_SENSORS):
#  for j in xrange(NUM_SENSORS + 1):
#    test_str = str(j + 1)
#    if j == NUM_SENSORS:
#      test_str = "ALL"
#
#    txt = r"\textbf{Multifold runs}: %d" % CONF_RUNS + '\n'
#    txt = txt + r"\textbf{Dataset size (train + test)}: %d" % dataset_size + '\n'
#    txt = txt + r"\textbf{T} - \textit{Time duration for averaging}" + '\n'
#    txt = txt + r"\textbf{C} - \textit{No. of independent variables}"
#
#
#    fig, ax = plotting.plot_violin(mapes_n[:, i, :, j].T,
#          title = r"$ NO_2 $\textbf{ MAPE: Training sensor %s, Testing %d}"\
#           % (test_str, i + 1), ylabel = r"\textit{MAPE} (\%)",
#          x_tick_labels = xlabs)
#
#    ax.annotate(txt, xy = (0.7, 0.8), xycoords='axes fraction')
#    fig.savefig(DIR_PREFIX + 'no2-train%stest%d.pdf' %\
#             (test_str, i+1), format='pdf')
#
#    fig, ax = plotting.plot_violin(mapes_o[:, i, :, j].T,
#          title = r"$ OX $\textbf{ MAPE: Training sensor %s, Testing %d}"\
#            % (test_str, i + 1), ylabel = r"\textit{MAPE} (\%)",
#          x_tick_labels = xlabs)
#
#    ax.annotate(txt, xy = (0.7, 0.8), xycoords='axes fraction')
#    fig.savefig(DIR_PREFIX + 'o3-train%stest%d.pdf' %\
#             (test_str, i+1), format='pdf')
#
# -------------------------------------------------------------------------------
#pages = pdfpublish.generate_text()
pdf = PdfPages(OUT_FILE_PREFIX + '-no2.pdf')

# print report data

# print figures
for (i, fig) in enumerate(no2_figs):
  text = 'Figure %d' % (i + 1)
  plt.text(0.05, 0.95, text, transform=fig.transFigure, size=10)
  pdf.savefig(fig)
  fig.savefig(DIR_PREFIX + no2_names[i] + '.png', format='png')
  plt.close(fig)

pdf.close()
del no2_figs, no2_names
gc.collect()
print 'NO2 PDF ready'

pdf = PdfPages(OUT_FILE_PREFIX + '-o3.pdf')

# print report data

# print figures
for (i, fig) in enumerate(o3_figs):
  text = 'Figure %d' % (i + 1)
  plt.text(0.05, 0.95, text, transform=fig.transFigure, size=10)
  pdf.savefig(fig)
  fig.savefig(DIR_PREFIX + o3_names[i] + '.png', format='png')
  plt.close(fig)

pdf.close()
del o3_figs, o3_names
gc.collect()
print 'O3 PDF ready'
# -------------------------------------------------------------------------------
# ---------------------- CORRELATE PM2.5 data -----------------------------------
# -------------------------------------------------------------------------------
if DEPLOYMENT > 1:
  aggregate_list = []

  aggregate_list.append(time_vec)
  aggregate_list.append(ref_pm25)

  if DEPLOY_SITE == 'MPCB':
    aggregate_list.append(ref_pm10)

  for i in xrange(NUM_SENSORS):
    aggregate_list.append(pm1[:, i].tolist())
    aggregate_list.append(pm25[:, i].tolist())
    aggregate_list.append(pm10[:, i].tolist())

  target_df = pd.DataFrame(aggregate_list).transpose()
  target_df = target_df.dropna()

  print "Data set size for PM (after dropna()): " + str(len(target_df.index))
  incl_pm10 = False
  if DEPLOY_SITE == 'MPCB':
    incl_pm10 = True

  figs, names = regress.pm_correlate(target_df, ref_pm10_incl=incl_pm10,
                                     loc_label=DEPLOY_SITE)

  pdf = PdfPages(OUT_FILE_PREFIX + '-pm.pdf')

  # print figures
  for (i, fig) in enumerate(figs):
    text = 'Figure %d' % (i + 1)
    plt.text(0.05, 0.95, text, transform=fig.transFigure, size=10)
    #fig.savefig(DIR_PREFIX + names[i] + '.pdf', format='pdf')
    pdf.savefig(fig)

  pdf.close()
  print 'PM PDF ready'
# -------------------------------------------------------------------------------
