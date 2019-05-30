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
  handles = []
  for y in y1.T:
    handle_1 = ax.scatter(epochs, y1, alpha=0.4, marker='.',
                          linewidths=0.2, color=colorWheel[ids[0]])
    handles.append(handle_1)
  for y in y2.T:
    handle_2 = ax2.scatter(epochs, y2, alpha=0.4, marker='.',
                           linewidths=0.2, color=colorWheel[ids[1]])
    handles.append(handle_2)

  # legend
  if leg_labels != None:
    ax.legend(handles=handles, labels=leg_labels,
              loc=3)

  return fig, ax
# ------------------------------------------------------------------------------
def plot_violin(X, title="", xlabel="", ylabel="", scale='auto',
                x_tick_labels=[], groups=None, group_labels=None,
                mark_maxoutliers=True, leg=False):
  '''
    Plot violins of different columns of X

    Params:
      X              - vector whose columns are plotted
      title          - Title of plot
      xlabel, ylabel - Axes labels
      scale          - 'auto' or list of two integers
      x_tick_labels  - tick labels for X
      groups         - list of integers
      group_labels   - labels for each group
      mark_maxoutliers - mark the maxima of violins cut off due to scale
                         on the top

    Return:
      fig, ax   - figure and axes with violins and labels
  '''

  islist = False
  if isinstance(X, list):
    if len(X) > 0:
      if not isinstance(X[0], list):
        X = np.array(X)
      else:
        islist = True
    else:
      return

  print islist

  if not islist and X.ndim == 1:
    X = X[:, np.newaxis]

  num_violins = len(X) if islist else X.shape[1]
  if groups is None:
    groups = [num_violins]
    group_labels = ['']
  else:
    if isinstance(groups, list):
      assert len(groups) == len(group_labels)
      assert np.sum(groups) == num_violins
    else:
      assert num_violins % groups == 0
      assert len(groups) == num_violins / groups
      groups = np.repeat(groups, num_violins/groups)

  even_groups = all([(group == groups[0]) for group in groups])
  label_size_match = np.size(x_tick_labels) == np.sum(groups)

  if len(x_tick_labels) == 0:
    for grsize in groups:
      x_tick_labels.append(range(1, grsize + 1))

  x_tick_labels = np.array(x_tick_labels)

  same_subgroups = False
  if not even_groups:
    assert label_size_match
  else:
    if not label_size_match:
      same_subgroups = True
      x_tick_labels = np.repeat(x_tick_labels[np.newaxis, :],
                                num_violins/groups[0], axis=0)

  # normal font size
  fs_norm = 12

  pos = []
  if islist:
    pos = range(0, len(X))
  elif X.ndim == 1:
    pos.append(1)
  else:
    for i in range(0, np.size(X, 1)):
      pos.append(2*i+1)

  means = []
  medians = []
  maxm = []

  if islist:
    for vio in X:
      means.append(np.mean(vio))
      medians.append(np.median(vio))
      maxm.append(np.max(vio))
  elif X.ndim == 1:
    means.append(np.mean(X))
    median.append(np.median(X))
    maxm.append(np.max(X))
  else:
    for data in X.T:
      means.append(np.mean(data))
      medians.append(np.median(data))
      maxm.append(np.max(data))

  print maxm
  # create figure
  fig=plt.figure(figsize=(10,8))
  ax = fig.add_subplot(111)

  # scale the plots
  if scale != 'auto':
    if scale == 'log':
      plt.yscale(scale)
    else:
      ax.set_ylim(scale)

  
  def autolabel():
    """
    Attach a text label above each bar displaying its height
    """
    pad = 1.20
    for i in range(0,len(pos)):
      va_mean = 'bottom'
      va_median = 'top'

      rotation ='0'
      fontsize=22
      if means[i] < medians[i]:
        va_mean = 'top'
        va_median = 'bottom'

      ht_median = medians[i]

      # bounds for height
      if scale != 'auto':
        if scale[0] + pad > ht_median:
          ht_median = scale[0] + pad
          va_median = 'bottom'
          rotation='90'
          fontsize=20
          
        if scale[1] < ht_median + pad:
          ht_median = scale[1] - pad
          va_median = 'top'
          rotation='90'
          fontsize=20

      #ax.text(pos[i]+0.1, means[i],
      #        '%.3g' % means[i],
      #        ha='center', va=va_mean, color='darkred', fontsize=15)
      ax.text(pos[i]+0.1, ht_median,
              '%.3g' % medians[i],
              ha='left', va=va_median, color='darkgreen',
              rotation=rotation, fontsize=fs_norm-3)

      if scale != 'auto' and mark_maxoutliers:
        if scale[1] < maxm[i] + pad and medians[i]  + pad < scale[1]:
          ax.text(pos[i]+0.20, scale[1] - pad,
                  '%.3g' % maxm[i],
                  ha='center', va='top', color='darkblue',
                  rotation='0', fontsize=fs_norm-5)

  #plt.tight_layout()
  plt.subplots_adjust(bottom=0.18, right=0.97, top=0.97, left=0.09)

  res = ax.violinplot(X, pos, points=20, widths=0.85,
             showmeans=True, showextrema=True, showmedians=True)

  autolabel()
  #x_tick_labels = ['Bandwidth (Mbits/sec)']

  # set colors for components
  lwidth = 0.85
  res['cmeans'].set(edgecolor='darkred', linewidth=1.2, label='mean')
  res['cmedians'].set(edgecolor='darkgreen', linewidth=1.2, label='median')
  res['cbars'].set(edgecolor='black', linewidth=lwidth, alpha=0.65)
  res['cmins'].set(edgecolor='black', linewidth=lwidth, alpha=0.75)
  res['cmaxes'].set(edgecolor='darkblue', linewidth=lwidth, alpha=0.75)

  if same_subgroups and len(groups) > 1:
    for i, lobe in enumerate(res['bodies']):
      lobe.set(color=colorWheel[(i % groups[0]) + 5], alpha=0.75)
  else:
    for i, lobe in enumerate(res['bodies']):
      lobe.set(alpha=0.55)

  
  # set ticklabels
  ax.set_xticks(pos)
  ax.set_xticklabels(x_tick_labels.flatten(), rotation='0')
  ax.tick_params(axis='y', labelsize=fs_norm-3)
  ax.tick_params(axis='x', labelsize=fs_norm-4)
  ax.tick_params(axis='y', which='minor', bottom=False)
  res = [t.set_y(0) for t in ax.get_xticklabels()]


  if scale != 'auto':
    if scale == 'log':
      plt.yscale(scale)
    else:
      ax.set_ylim(scale)


  # set group labels on x axis
  pos = np.zeros(2, dtype=int)
  loc = ax.get_xticks()
  bottom, top = ax.get_ylim()
  axloc = ax.get_position().bounds[:2]
  corr_align = [.01, -.03, -.03]
  #ticklab_pos = ax.get_xticklabels()[0].get_position()[1]
  #ticklab_pos += (ax.get_ylim()[0] - ax.get_ylim()[1]) / 20
  for (group, gl, corr) in zip(groups, group_labels, corr_align):
    pos[0] = pos[1]
    pos[1] = pos[0] + group

    set_break = False
    if pos[1] >= len(loc):
      pos[1] -= 1
      set_break = True

    #print ticklab_pos
    x = (2*axloc[0] + loc[pos[0]] + loc[pos[1]]) / (2.0 * (loc[-1] - loc[0]) + 4)
    #gl = r'\textbf{%s}' % gl
    plt.figtext(x+corr, 0.005, gl, ha='center', va='bottom', fontsize=fs_norm-3)

    if set_break:
      break

    # boundary for each group
    line_loc = (loc[pos[1]] + loc[pos[1] - 1]) / 2
    bdy_line = Line2D([line_loc, line_loc], [top, bottom], color='black',
                      lw=1.2, linestyle='-')
    ax.add_line(bdy_line)


  #if X.ndim > 1:
  #  ax.set_xlim([0,2*np.size(X, 1)])
  #  legend = ax.legend(loc='upper right', shadow=True, fontsize='small')

  ax.set_xlabel(xlabel, fontsize=fs_norm-2)
  ax.set_ylabel(ylabel, fontsize=fs_norm-2)
  ax.set_title(title, fontsize=fs_norm)

  custom_lines = [Line2D([0], [0], color='darkred', lw=1.2),
                  Line2D([0], [0], color='darkgreen', lw=1.2),
                  Line2D([0], [0], color='darkblue', lw=1.2)]

  if leg == True:
    ax.legend(handles=custom_lines, labels=['mean', 'median', 'max'], shadow=False,
              fontsize=fs_norm-4,ncol=1,loc=(0.01, 0.01), framealpha=0.65)

  plt.grid()
  plt.minorticks_on()
  ax.yaxis.grid(b=True, which='minor', color='gray', linestyle='--', lw=0.5, alpha=0.5)
  ax.yaxis.grid(b=True, which='major', color='gray', linestyle='-', lw=0.8, alpha=1)
  ax.xaxis.grid(b=True, which='major', color='gray', linestyle='-', lw=0.8, alpha=1)

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

  fig, ax = plot_stem(mean, mean_err,
              title=title, xlabel=xlabel,
              ylabel=r'\textit{Mean Absolute Percentage Error}')
 
  for i in xrange(num_bins):
    y = min(mean_err[i] + pad, ax.get_ylim()[1])
    ax.annotate('%d' % lens[i], xy=(mean[i], y), rotation = 80, alpha=0.8,
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
