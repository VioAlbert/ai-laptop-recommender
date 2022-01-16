import csv
import numpy as np

# Input CSV

with open('data.csv') as csv_file:
  laptops = list(csv.reader(csv_file, delimiter=','))
  laptops.pop(0)
  count_spec = len(laptops[0]) - 1
  for i in range(1, count_spec+1):
    for row in laptops:
      row[i] = float(row[i])

laptops = np.array(laptops, dtype='object')

# Normalize each parameter

mn = np.min(laptops, axis=0)
mx = np.max(laptops, axis=0)
ram_sizes = np.unique(laptops[:,2]) # for debugging purposes

for row in laptops:
  for i in range(1, count_spec+1):
    row[i] = (row[i] - mn[i]) / (mx[i] - mn[i])

with open('data_normalized.csv', 'w', encoding='UTF8') as csv_file:
  writer = csv.writer(csv_file)
  for row in laptops:
    writer.writerow(row)
