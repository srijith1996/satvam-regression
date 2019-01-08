# Regression scripts
## for SATVAM gas sensor calibration
This is a simple utility for regression of SATVAM gas sensor data. Data is ingested by
specifying paths to the files containing the reference monitor data and the data from 
the SATVAM gas sensors. Time alignment is taken care of by the script.

### Dependencies
matplotlib >= 3.0.2 <br\>
sklearn <br\>
numpy

### Usage

Run the following command on a BASH terminal:

```sh

  python format-csv.py ref-monitor.xlsx ebam.csv sens1.csv sens2.csv sens3.csv outfile
    # replace ref-monitor.xlsx with Excel file directly downloaded
    # from the reference monitor and ebam.csv with the EBAM data file
    # replace sens*.csv with CSV files of the sensors
    # directly downloaded from Graphana
    # replace outfile with the output file prefix

```

The output PDFs will be generated starting with the prefix specified in the
command above, one each for NO2, O3 and PM2.5 inference plots
