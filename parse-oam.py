import sys 
import plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

infile = sys.argv[1]
outfile = sys.argv[2]

node = dict()
link = dict()
epochs = dict()

def parse_node_data(line, epoch):
  line = line.strip()
  words = line.split(' ')

  node_id = words[0]
  node_temp = float(words[2][:-1]) / 1000
  node_volt = float(words[5][:-3]) / 1000

  if not node_id in node:
    epochs[node_id] = []
    node[node_id] = []

  node[node_id].append(dict())
  node[node_id][len(node[node_id]) - 1]["t"] = node_temp
  node[node_id][len(node[node_id]) - 1]["bv"] = node_volt

  epochs[node_id].append(epoch)

  return node_id

def parse_link_data(line, child_id, pt_id, epoch):
  line = line.strip()
  words = line.split(' ')
  
  link_id = '%s-%s' % (child_id, pt_id)
  link_rssi = float(words[2][:-3])
  link_fd = float(words[4][:-1])

  if not link_id in link:
    epochs[link_id] = []
    link[link_id] = []

  link[link_id].append(dict())
  link[link_id][len(link[link_id]) - 1]["rssi"] = link_rssi
  link[link_id][len(link[link_id]) - 1]["fd"] = link_fd

  epochs[link_id].append(epoch)

  return link_id

with open(infile, 'r') as fh:
  start_epoch = float(fh.readline().split(' ')[2])

  start = True
  while True:
    # skip empty line
    if start:
      fh.readline()
      start = False

    line = fh.readline()
    if not line:
      break

    diff_epoch = float(line.strip()[1:-1])
    epoch = int(start_epoch + diff_epoch)

    just_entered = True
    first_line = True
    while True: 
      # check if end of link is reached
      if (not just_entered) and fh.readline()[0] != '|':
        just_entered = True
        break
      else:
        just_entered = False

      if first_line:
        line = fh.readline()
        node_name = parse_node_data(line, epoch)
        first_line = False

        # skip pipe
        fh.readline()

      link_line = fh.readline()
      
      # skip pipe
      fh.readline()

      line = fh.readline()
      pt_node_name = parse_node_data(line, epoch)

      parse_link_data(link_line, node_name, pt_node_name, epoch)

      node_name = pt_node_name

 
print "Following nodes and links have existed in the past run"
for key in epochs.keys():
  print key

figs = []
for key in node.keys():
  temps = []
  volts = []
  for entry in node[key]:
    temps.append(entry['t'])
    volts.append(entry['bv'])
  
  fig, ax = plotting.ts_plot(epochs[key], temps,
        title = ('%s Temperature characteristic' % key),
        ylabel = r'Temperature ($ ^{\circ} C $)',
        leg_labels=[key])

  figs.append(fig)

  fig, ax = plotting.ts_plot(epochs[key], volts,
        title = ('%s Battery Voltage characteristic' % key),
        ylabel = r'Voltage (mV)', ylim=[2.5, 4.0],
        leg_labels=[key])

  figs.append(fig)

for key in link.keys():
  rssis = []
  fds = []
  for entry in link[key]:
    rssis.append(entry['rssi'])
    fds.append(entry['fd'])
  
  fig, ax = plotting.ts_plot(epochs[key], rssis,
        title = ('%s RSSI characteristic' % key),
        ylabel = r'RSSI (dB)',
        leg_labels=[key])

  figs.append(fig)

  fig, ax = plotting.ts_plot(epochs[key], fds,
        title = ('%s Frames dropped characteristic' % key),
        ylabel = r'Frames dropped',
        leg_labels=[key])

  figs.append(fig)

# save figures to pdf
pdf = PdfPages(outfile)
for fig in figs:
  pdf.savefig(fig)
pdf.close()
