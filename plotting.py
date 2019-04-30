# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator, MultipleLocator
import matplotlib as mpl
from matplotlib import rc
import time

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
# ------------------------------------------------------------------------------
# Load style file
plt.close('all')
plt.style.use('paperDoubleFig.mplstyle')
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# Make some style choices for plotting
colorWheel =['#329932',
            '#ff6961',
            'b',
            '#6a3d9a',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
            '#67001f',
            '#b2182b',
            '#d6604d',
            '#f4a582',
            '#fddbc7',
            '#f7f7f7',
            '#d1e5f0',
            '#92c5de',
            '#4393c3',
            '#2166ac',
            '#053061']
dashesStyles = [[3,1],
            [1000,1],
            [2,1,10,1],
            [4, 1, 1, 1, 1, 1]]

# utilities
#def axesDimensions(ax):
#    if hasattr(ax, 'get_zlim'): 
#        return 3
#    else:
#        return 2
# ------------------------------------------------------------------------------
def set_plot_labels(ax, title='', xlabel='', ylabel=''):

  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)

# ------------------------------------------------------------------------------
def plot_reg_plane(X, Y, coeffs, points=40,
                   title='', xlabel='', ylabel='',
                   zlabel='', axLim=None):
  '''
    Plot the surface or line that is the result of regression
  
    Params:
      X       : 2d array with features as columns
      y       : 2d array with outputs
      coeffs  : array of coeffs [constant, x, y]
      points  : number of points to generate for line/plane
      title   : Title string of the plot
      xlabel  : X axis label of the plot
      ylabel  : Y axis label of the plot
      zlabel  : Z axis label of the plot
      axLim   : Limits of the dependent axis

    Return:
      fig     : figure plotted
  '''

  # check for errors
  if np.size(coeffs) < 1 or np.size(coeffs) > 3:
    print "Too few or too many coeffs to visualize in 3d"
    return -1

  if len(np.shape(X)) == 1:
    tmp_X = np.reshape(X, [len(X), 1])

    if np.size(coeffs) != np.shape(tmp_X)[1] + 1:
      print "coeffs and dimensions of X don't match"
      return -1

  # create plot figure
  fig = plt.figure()

  # plot the line
  if np.size(coeffs) == 2:
    ax = fig.add_subplot(111)
    x = np.linspace(np.min(X), np.max(X), points)
    y = coeffs[0] + x * coeffs[1]

    ax.plot(x, y, color=colorWheel[0], alpha=0.6)
    ax.scatter(X, Y, color=colorWheel[1],
               alpha=0.7, marker='.',
               linewidths=0.3)

  else:
    ax = fig.add_subplot(111, projection='3d')
    print np.min(X[:, 0]), np.min(X[:, 1])
    print np.max(X[:, 0]), np.max(X[:, 1])
    x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), points)
    y = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), points)
    X_grid, Y_grid = np.meshgrid(x, y)
    Z_grid = coeffs[0] + X_grid * coeffs[1] + Y_grid * coeffs[2]

    ax.plot_wireframe(X_grid, Y_grid, Z_grid, color=colorWheel[0],
                      alpha=0.3)

    # plot the scatter of data
    ax.scatter(X[:,0], X[:,1], Y,
               color=colorWheel[1], alpha=0.7,
               marker='.', linewidths=0.3)

  # set axis properties
  set_plot_labels(ax, title, xlabel, ylabel)

  if np.size(coeffs) == 3:
    ax.set_zlabel(zlabel)

  if axLim != None:
    ax.set_zlim(-axLim, axLim)

  return fig
# ------------------------------------------------------------------------------
def format_date(x, pos=None):
  return time.strftime('%d/%m %H:%M', time.localtime(x))
# ------------------------------------------------------------------------------
def format_x_date(fig, ax):
  fig.autofmt_xdate()

  majorLocator = MultipleLocator(604800)  # week boundary
  minorLocator = MultipleLocator(86400)   # day boundary

  ax.xaxis.set_major_locator(majorLocator)
  ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
  ax.xaxis.set_minor_locator(minorLocator)

  ax.grid(b=True, which='major', axis='both',
          color='gray', linestyle='--', alpha=0.5)
  
  ax.grid(b=True, which='minor', axis='both',
          color='gray', linestyle=':', alpha=0.3)
  
  ax.set_xlabel(r'\textbf{Time}')
  plt.xticks(rotation=30)
# ------------------------------------------------------------------------------
def plot_xy(x, y, title='', xlabel='', ylabel='', ylim=None, leg_labels=None):
  '''
    Plot on X-Y plane
  '''

  fig = plt.figure()
  ax = fig.add_subplot('111')

  set_plot_labels(ax, title, xlabel, ylabel)
  if ylim is not None:
    ax.set_ylim(ylim)

  ax.set_xticks(x)
  ax.grid(b=True, which='major', axis='both',
          color='gray', linestyle='--', alpha=0.5)

  ax.grid(b=True, which='minor', axis='both',
          color='gray', linestyle=':', alpha=0.3)

  if len(y.shape) == 1:
    y = np.reshape(y, [np.size(y), 1])

  for (i, col) in enumerate(y.T):
    ax.plot(x, col, alpha=1.0, linewidth=1.5, color=colorWheel[i+2])

  if leg_labels is not None:
    ax.legend(labels=leg_labels, loc=3)

  return fig, ax
# ------------------------------------------------------------------------------
def ts_plot(epochs, Y, title='', ylabel='',
            ylim=None, leg_labels=None, ids=[]):
  '''
     Plot a time-series of columns of Y

     Params:
        epochs    : Timestamp epochs as integers
        Y         : ndarray with axis=1 as sequences
        title     : Plot title
        ylabel    : y-axis label
        ylim      : list of y axis limits [min, max]
        leg_labels: legend labels as list

     Return:
        fig - matplotlib figure
  '''

  fig = plt.figure()
  ax = fig.add_subplot('111')

  set_plot_labels(ax, title, xlabel='', ylabel=ylabel)
  format_x_date(fig, ax)

  if ylim != None:
    ax.set_ylim(ylim)

  if ids == []:
    ids = range(1, np.size(Y, axis=1)+1)

  handles = []
  if np.size(Y) == np.shape(Y)[0]:
    ax.plot(epochs, Y, alpha=0.6, linewidth=1.0, color=colorWheel[ids[0]])
    handles.append(mpatches.Patch(color=colorWheel[ids[0]]))

    mean_line = Line2D([epochs], np.repeat(np.mean(Y), np.size(epochs)),
                       linewidth=1.0, color=colorWheel[ids[0]])
    ax.add_line(mean_line)

  else:
    for (i, col) in enumerate(Y.T):
      ax.plot(epochs, col, alpha=0.6,
            linewidth=0.7, color=colorWheel[ids[i]])
      handles.append(mpatches.Patch(color=colorWheel[ids[i]]))

      mean_line = Line2D([epochs], np.repeat(np.mean(col), np.size(epochs)),
                       linewidth=1.0, color=colorWheel[ids[i]])
      ax.add_line(mean_line)

  # legend
  if leg_labels != None:
    ax.legend(handles=handles, labels=leg_labels, loc=3)

  return fig, ax
# ------------------------------------------------------------------------------
def compare_ts_plot(epochs, y1, y2, title='', ylabel='',
                    ylabel_s='', ylim_p=None, ylim_s=None,
                    leg_labels=None, ids=[]):
  '''
     Plot a time-series on twin axes comparing the y1 and y2

     Params:
      epochs  : Timestamp epochs as integers
      y1      : 1st sequence
      y2      : 2nd sequence
      title   : Title string of the plot
      ylabel  : Y axis label of the plot
      ylabel_s: secondary Y axis label of the plot
      ylim_p  : set y limits for primary axis
      ylim_s  : set y limits for secondary axis
      legend
      
     Return:
      fig  : plot figure handle
      ax   : plot axis handle
  '''

  fig = plt.figure()
  ax = fig.add_subplot('111')
  ax2 = ax.twinx()

  set_plot_labels(ax, title, xlabel='', ylabel=ylabel)
  set_plot_labels(ax2, title='', xlabel='', ylabel=ylabel_s)
  format_x_date(fig, ax)

  if ylim_p != None:
    ax.set_ylim(ylim_p)
  if ylim_s != None:
    ax2.set_ylim(ylim_s)

  if ids == []:
    ids = range(1, np.size(Y, axis=1)+1)

  # plot residuals and temperatures
  handle_1 = ax.scatter(epochs, y1, alpha=0.4, marker='.',
                        linewidths=0.2, color=colorWheel[ids[0]])
  handle_2 = ax2.scatter(epochs, y2, alpha=0.4, marker='.',
                         linewidths=0.2, color=colorWheel[ids[1]])

  # legend
  if leg_labels != None:
    ax.legend(handles=[handle_1, handle_2], labels=leg_labels,
              loc=3)

  return fig, ax
# ------------------------------------------------------------------------------
def plot_violin(X, title="violin plot", xlabel="", ylabel="", scale='auto',
                x_tick_labels=[]):

  pos = []
  if X.ndim == 1:
    pos.append(1)
  else:
    for i in range(0, np.size(X, 1)):
      pos.append(2*i+1)

  means = []
  medians = []

  if X.ndim == 1:
    means.append(np.mean(X))
    medians.append(np.median(X))
  else:
    for data in X.T:
      means.append(np.mean(data))
      medians.append(np.median(data))

  def autolabel():
    """
    Attach a text label above each bar displaying its height
    """
    for i in range(0,len(pos)):
      va_mean = 'bottom'
      va_median = 'top'

      if means[i] < medians[i]:
        va_mean = 'top'
        va_median = 'bottom'

      ax.text(pos[i]+0.20, means[i],
              '%.4f' % means[i],
              ha='center', va=va_mean, color='red')
      ax.text(pos[i]+0.20, medians[i],
              '%.4f' % medians[i],
              ha='center', va=va_median,color='green')

  fig=plt.figure(figsize=(10,7))
  ax = fig.add_subplot(111)
  
  plt.grid()
  ax.yaxis.grid(b=True, which='minor', color='g', linestyle='-', alpha=0.2)
  #plt.minorticks_on()

  plt.tick_params(labelsize=13)

  parts = ax.violinplot(X, pos, points=20, widths=0.4,
             showmeans=True, showextrema=True, showmedians=True)

  autolabel()
  #x_tick_labels = ['Bandwidth (Mbits/sec)']
  parts['cmeans'].set_edgecolor('darkred')
  parts['cmedians'].set_edgecolor('darkgreen')

  ax.set_xticks(pos)
  
  # set ticklabels
  if len(x_tick_labels) == 0:
    ax.set_xticklabels([(x + 1)/2 for x in pos])
  else:
    ax.set_xticklabels(x_tick_labels, rotation='0', fontsize=13)

  if X.ndim > 1:
    ax.set_xlim([0,2*np.size(X, 1)])
    legend = ax.legend(loc='upper right', shadow=True, fontsize='small')

  ax.set_xlabel(xlabel, fontsize=15)
  ax.set_ylabel(ylabel, fontsize=15)
  ax.set_title(title, fontsize=15)

  if scale != 'auto':
    ax.set_ylim(scale)

  custom_lines = [Line2D([0], [0], color='red', lw=1.5),
                Line2D([0], [0], color='green', lw=1.5)]

  ax.legend(custom_lines, ['Mean', 'Median'],shadow='True',
            fontsize=13,ncol=1,loc='upper left')  

  return fig, ax
# ------------------------------------------------------------------------------
def plot_stem(x, y, title='', xlabel='', ylabel='', ylim=100):
  
  fig = plt.figure()
  ax = fig.add_subplot('111')

  set_plot_labels(ax, title, xlabel, ylabel)

  ax.stem(x, y, linestyle='-', alpha=1.0, marker='o')
  ax.set_ylim([0, ylim])

  plt.grid()
  plt.minorticks_on()
  ax.yaxis.grid(b=True, which='major', linestyle='-', alpha=0.8)
  ax.yaxis.grid(b=True, which='minor', linestyle=':', alpha=0.5)
  ax.xaxis.grid(b=True, which='major', linestyle='-', alpha=0.8)
  ax.xaxis.grid(b=True, which='minor', linestyle=':', alpha=0.5)

  return fig, ax
# ------------------------------------------------------------------------------
def plot_err_dist(true_val, pred, num_bins=40, pad=0.3,
                  title='', xlabel=''):
  '''
    Plot the distribution of error w.r.t true vals
  '''

  err = np.abs((pred - true_val)/true_val) * 100
  bin_edges = np.arange(0, num_bins + 1) * np.max(true_val)/num_bins

  mean_err = np.zeros([num_bins, ])
  mean = np.zeros([num_bins, ])
  lens = np.zeros([num_bins, ])

  for i in xrange(len(bin_edges)-1):
    mean_err[i] = np.mean(err[[(val >= bin_edges[i]
                            and val < bin_edges[i+1]) for val in true_val]])
    mean[i] = np.mean(true_val[[(val >= bin_edges[i]
                            and val < bin_edges[i+1]) for val in true_val]])
    lens[i] = np.size(true_val[[(val >= bin_edges[i]
                            and val < bin_edges[i+1]) for val in true_val]])

  fig, ax = plotting.plot_stem(mean, mean_err,
              title=title, xlabel=xlabel,
              ylabel=r'\textit{Mean Absolute Percentage Error}')

  for i in xrange(num_bins):
    ax.annotate('%d' % lens[i], xy=(mean[i], mean_err[i] + pad), rotation = 80, alpha=0.8,
                fontsize='small')

  return fig
  
# ------------------------------------------------------------------------------
def plot_hist(ax, v, bins=20, title='', ids=None):
  '''
    Plot the histogram of samples in v
  '''

  if np.size(v) == v.shape[0]:
    v = np.reshape(v, [np.size(v), 1])

  if ids == None:
    ids = xrange(v.shape[1])

  for (i, col) in enumerate(v.T):
    n, out_bins, patches = ax.hist(v[:,i], bins, density=True,
         facecolor=colorWheel[ids[i]], alpha=0.6)
  set_plot_labels(ax, title='', xlabel=title, ylabel='')

# ------------------------------------------------------------------------------
def inset_hist_fig(fig, outer_ax, v, size, loc, ids=None):
  '''
    Inset histogram into the outer_ax at location.

    Params:
      fig       - figure
      outer_ax  - Main axes
      v         - array of values
      size      - percent of parent fig size [width, height]
      loc       - quadrant location

    Return:
      fig - The main figure with inset figure
  '''

  in_axes = inset_axes(outer_ax, width=size[0],
          height=size[1], loc=loc)

  #in_axes.patch.set_alpha(0.6)

  plot_hist(in_axes, v, bins=200, title=r'Distribution', ids=ids)

  return fig
# ------------------------------------------------------------------------------
