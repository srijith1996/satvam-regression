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

    use [<ebam_file>] only when DEPLOY_SITE is set to 'mru'

'''
# -------------------------------------------------------------------------------
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
# -------------------------------------------------------------------------------
# Change this field to 'mpcb' or 'mru' based on deployment site
DEPLOY_SITE = 'mru'

DEPLOY_SITE = DEPLOY_SITE.lower()

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

SENS_TS_FORMAT = '%Y-%m-%dT%H:%M:%S.000Z'  # Input timestamp format
DES_FORMAT = '%Y/%m/%d %H:%M:%S'      # Output format

#reference monitor data fields
if DEPLOY_SITE == 'mru':

  # input arg files
  REF_FILE = sys.argv[1]
  EBAM_FILE = sys.argv[2]

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

  # EBAM data fields
  EBAM_HEADER_ROW = 0
  EBAM_TS_FIELD_HDR = 'Time'
  EBAM_PM25_FIELD_HDR = 'ConcRT (mg/m3)'

  SENSOR_FILE_LIST = sys.argv[3:-1]

elif DEPLOY_SITE == 'mpcb':

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
# ---------------- PRE-PROCESS SATVAM SENSOR DATA -------------------------------
# -------------------------------------------------------------------------------
print "Processing SATVAM data........"

# list of dataframes for each mote
sens_dfs = []
min_time = []
max_time = []

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

  sens_dfs[i] = sens_dfs[i][sens_dfs[i].applymap(lambda x: 
            (x != "NoData" and x != "NO_DATA"
         and x != "undefined")).all(1)].dropna()

  print "Sensor %d has %d points" % (i, len(sens_dfs[i].index))

  min_time.append(timestamps[0])
  max_time.append(timestamps[-1])

# find the minimum and maximum epochs
min_time = np.min(min_time)
max_time = np.max(max_time)
#min_time -= min_time % 60
#max_time -= (max_time % 60) + 60

print "DONE"
# -------------------------------------------------------------------------------
# ----------------- PRE-PROCESS REFERENCE MONITOR DATA --------------------------
# -------------------------------------------------------------------------------
print "Processing Reference monitor data........"

ref_df = pd.read_excel(REF_FILE, header=R_HEADER_ROW, skiprows=R_SKIP_ROWS)
ref_df = ref_df.drop(np.arange(len(ref_df) - R_SKIP_ROWS_END, len(ref_df)))

ref_df = ref_df[ref_df.applymap(lambda x:
            (x != "NoData" and x != "NO_DATA"
         and x != "RS232" and x != "CALIB_S"
         and x != "CALIB_Z" and x != "FAULTY"
         and x != "Samp<" and x != "MAINT")).all(1)].dropna()

# clean up time values that are 24:00
for index, row in ref_df.iterrows():
  if row[R_TIME_FIELD_HDR] == '24:00' or row[R_TIME_FIELD_HDR][-5:] == '24:00':
    if DEPLOY_SITE == 'mru':
      row[R_TIME_FIELD_HDR] = '00:00'
      row[R_DATE_FIELD_HDR] = dt.datetime.strptime(
          row[R_DATE_FIELD_HDR], REF_DATE_FORMAT) + timedelta(days=1)
      row[R_DATE_FIELD_HDR] = dt.datetime.strftime(row[R_DATE_FIELD_HDR],
          REF_DATE_FORMAT)

    elif DEPLOY_SITE == 'mpcb':
      row[R_TIME_FIELD_HDR] = row[R_TIME_FIELD_HDR][:-5] + '00:00'
      row[R_TIME_FIELD_HDR] = dt.datetime.strptime(
          row[R_TIME_FIELD_HDR], REF_TS_FORMAT) + timedelta(days=1)
      row[R_TIME_FIELD_HDR] = dt.datetime.strftime(row[R_TIME_FIELD_HDR],
          REF_TS_FORMAT)

print "Interpreting time stamps...."
times = ref_df[R_TIME_FIELD_HDR].values

if DEPLOY_SITE == 'mru':
  dates = ref_df[R_DATE_FIELD_HDR].values

for i in xrange(len(times)):

  if DEPLOY_SITE == 'mru':
    times[i] = dates[i] + ' ' + times[i]

  times[i] = dt.datetime.strptime(times[i], REF_TS_FORMAT).strftime('%s')
  times[i] = int(times[i])

  # change resolution to minutes
  times[i] -= times[i] % 60

print "Reference monitor has %d points" % len(ref_df.index)

if DEPLOY_SITE == 'mru':
  ref_df = ref_df.drop(columns=[R_DATE_FIELD_HDR])

min_time = min([times[0], min_time])
max_time = max([times[-1], max_time])

#print min_time, max_time

print "DONE"
#print ref_df
# -------------------------------------------------------------------------------
# ----------------- PRE-PROCESS EBAM DATA ---------------------------------------
# -------------------------------------------------------------------------------
if DEPLOY_SITE == 'mru':
  print "Processing EBAM data........"
  
  ebam_df = pd.read_csv(EBAM_FILE, header=EBAM_HEADER_ROW)
  
  #ebam_df = ebam_df[ebam_df.applymap(lambda x:
  #            (x != "NoData" and x != "NO_DATA"
  #         and x != "RS232" and x != "CALIB_S"
  #         and x != "CALIB_Z" and x != "FAULTY"
  #         and x != "Samp<")).all(1)].dropna()
  
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
  #print ref_df
# -------------------------------------------------------------------------------
# generate the time vector
time_vec = np.arange(min_time, max_time+60, 60)
#print time.strftime(DES_FORMAT, time.gmtime(time_vec[-1]))

# copy data for each sensor 
no2_op1 = np.empty([len(time_vec), NUM_SENSORS])
no2_op2 = np.empty([len(time_vec), NUM_SENSORS])
ox_op1 = np.empty([len(time_vec), NUM_SENSORS])
ox_op2 = np.empty([len(time_vec), NUM_SENSORS])
temp = np.empty([len(time_vec), NUM_SENSORS])
#hum = np.empty([len(time_vec), NUM_SENSORS])
pm1 = np.empty([len(time_vec), NUM_SENSORS])
pm25 = np.empty([len(time_vec), NUM_SENSORS])
pm10 = np.empty([len(time_vec), NUM_SENSORS])

no2_op1[:] = no2_op2[:] = ox_op1[:] = ox_op2[:] = temp[:] = np.nan
pm1[:] = pm25[:] = pm10[:] = np.nan

print "Time-stamp aligning SATVAM sensor values...."
for i in xrange(NUM_SENSORS):
  
  print "Sensor " + str(i + 1)
  # collect values
  sens_ts = sens_dfs[i][TIME_FIELD_HDR].values
  sens_no2op1 = sens_dfs[i][NO2OP1_FIELD_HDR].values
  sens_no2op2 = sens_dfs[i][NO2OP2_FIELD_HDR].values
  sens_oxop1 = sens_dfs[i][OXOP1_FIELD_HDR].values
  sens_oxop2 = sens_dfs[i][OXOP2_FIELD_HDR].values
  sens_temp = sens_dfs[i][TEMP_FIELD_HDR].values
  #sens_hum = sens_dfs[i][HUM_FIELD_HDR].values
  sens_pm1 = sens_dfs[i][PM1_FIELD_HDR].values
  sens_pm25 = sens_dfs[i][PM25_FIELD_HDR].values
  sens_pm10 = sens_dfs[i][PM10_FIELD_HDR].values

  # align to time_vec
  for j in xrange(len(sens_ts)):
    ts_index = time_vec.tolist().index(sens_ts[j])
    no2_op1[ts_index, i] = sens_no2op1[j]
    no2_op2[ts_index, i] = sens_no2op2[j]
    ox_op1[ts_index, i] = sens_oxop1[j]
    ox_op2[ts_index, i] = sens_oxop2[j]
    temp[ts_index, i] = sens_temp[j]

    pm1[ts_index, i] = sens_pm1[j]
    pm25[ts_index, i] = sens_pm25[j]
    pm10[ts_index, i] = sens_pm10[j]

#print no2_op1
print "DONE"
# -------------------------------------------------------------------------------
print "Time-stamp aligning ref monitor data...."

ref_ts = ref_df[R_TIME_FIELD_HDR].values
ref_no2 = np.empty([len(time_vec), ])
ref_o3 = np.empty([len(time_vec), ])

ref_pm10 = np.empty([len(time_vec), ])
ref_pm25 = np.empty([len(time_vec), ])

ref_no2[:] = ref_o3[:] = np.nan
ref_pm25[:] = ref_pm10[:] = np.nan

for j in xrange(len(ref_ts)):
  ts_index = time_vec.tolist().index(ref_ts[j])
  ref_no2[ts_index] = ref_df[R_NO2_FIELD_HDR].values[j]
  ref_o3[ts_index] = ref_df[R_OX_FIELD_HDR].values[j]

  if DEPLOY_SITE == 'mpcb':
    ref_pm10[ts_index] = ref_df[R_PM10_FIELD_HDR].values[j]
    ref_pm25[ts_index] = ref_df[R_PM25_FIELD_HDR].values[j]

print "DONE"
#print "Sizes:"
#print np.shape(time_vec)
#print np.shape(no2_op1)
#print np.shape(no2_op2)
#print np.shape(ox_op1)
#print np.shape(ox_op2)
#print np.shape(temp)
#print np.shape(ref_no2)
#print np.shape(ref_o3)
# -------------------------------------------------------------------------------
if DEPLOY_SITE == 'mru':
  print "Time-stamp aligning EBAM data..."
  
  # ebam currently measures only pm2.5
  ebam_ts = ebam_df[EBAM_TS_FIELD_HDR].values
  
  for j in xrange(len(ebam_ts)):
    ts_index = time_vec.tolist().index(ebam_ts[j])
    
    # convert to ug/m3
    ref_pm25[ts_index] = ebam_df[EBAM_PM25_FIELD_HDR].values[j] * 1000
  
  print "DONE"
# -------------------------------------------------------------------------------
aggregate_list = []

aggregate_list.append(time_vec)
aggregate_list.append(ref_no2)
aggregate_list.append(ref_o3)

for i in xrange(NUM_SENSORS):
  aggregate_list.append(temp[:, i].tolist())
  aggregate_list.append(no2_op1[:, i].tolist())
  aggregate_list.append(no2_op2[:, i].tolist())
  aggregate_list.append(ox_op1[:, i].tolist())
  aggregate_list.append(ox_op2[:, i].tolist())

target_df = pd.DataFrame(aggregate_list).transpose()
target_df = target_df.dropna()
print "Data set size (after dropna()): " + str(len(target_df.index))
# -------------------------------------------------------------------------------
print "Calling regression algorithm on obtained DataFrame"
no2_figs, no2_fignames, o3_figs, o3_fignames = regress.regress_df(target_df, runs=5000)
#pages = pdfpublish.generate_text()

pdf = PdfPages(OUT_FILE_PREFIX + '-no2.pdf')

# print report data

# print figures
for (i, fig) in enumerate(no2_figs):
  text = 'Figure %d' % (i + 1)
  plt.text(0.05, 0.95, text, transform=fig.transFigure, size=10)
  pdf.savefig(fig)
  fig.savefig(DIR_PREFIX + no2_fignames[i], format='svg')
  plt.close(fig)

pdf.close()
print 'NO2 PDF ready'

pdf = PdfPages(OUT_FILE_PREFIX + '-o3.pdf')

# print report data

# print figures
for (i, fig) in enumerate(o3_figs):
  text = 'Figure %d' % (i + 1)
  plt.text(0.05, 0.95, text, transform=fig.transFigure, size=10)
  pdf.savefig(fig)
  fig.savefig(DIR_PREFIX + o3_fignames[i], format='svg')
  plt.close(fig)

pdf.close()
print 'O3 PDF ready'
# -------------------------------------------------------------------------------
# ---------------------- CORRELATE PM2.5 data -----------------------------------
# -------------------------------------------------------------------------------
aggregate_list = []

aggregate_list.append(time_vec)
aggregate_list.append(ref_pm25)

for i in xrange(NUM_SENSORS):
  aggregate_list.append(pm1[:, i].tolist())
  aggregate_list.append(pm25[:, i].tolist())
  aggregate_list.append(pm10[:, i].tolist())

target_df = pd.DataFrame(aggregate_list).transpose()
target_df = target_df.dropna()

print "Data set size for PM (after dropna()): " + str(len(target_df.index))
# -------------------------------------------------------------------------------
figs, names = regress.pm_correlate(target_df)

pdf = PdfPages(OUT_FILE_PREFIX + '-pm.pdf')

# print figures
for (i, fig) in enumerate(figs):
  text = 'Figure %d' % (i + 1)
  plt.text(0.05, 0.95, text, transform=fig.transFigure, size=10)
  fig.savefig(DIR_PREFIX + names[i], format='svg')
  pdf.savefig(fig)

pdf.close()
print 'PM PDF ready'
# -------------------------------------------------------------------------------