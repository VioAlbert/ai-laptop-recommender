import csv
import numpy as np

# variables

count_spec = 7
query = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

# adjustable variables

threshold = 1
c_dist = 3
c_sim = 5

# Input CSV

with open('data.csv') as csv_file:
  laptops = list(csv.reader(csv_file, delimiter=','))
  for i in range (1, count_spec+1):
    for row in laptops:
      row[i] = float(row[i])

laptops = np.array(laptops, dtype='object')

# Normalize each parameter

mn = np.min(laptops, axis=0)
mx = np.max(laptops, axis=0)

for row in laptops:
  for i in range(1, count_spec+1):
    row[i] = (row[i] - mn[i]) / (mx[i] - mn[i])
  row[1] = 1 - row[1] # for processors, the smaller the value, the better

print(laptops)
print()

# Find similarity using euclidean distance and cosine similarity

nearest = np.empty((0, 2), dtype='object')

for row in laptops:
  dist = np.linalg.norm(query - row[1:8])
  sim = query.dot(row[1:8]) / (np.linalg.norm(query) * np.linalg.norm(row[1:8]))
  val = (c_dist * dist + c_sim * sim) / (c_dist + c_sim)
  nearest = np.append(nearest, np.array([[row[0], val]], dtype='object'), axis=0)

nearest = nearest[np.argsort(nearest[:,1])]

nearest = nearest[nearest[:,1] < threshold]

print(f'Nearest laptops:\n{nearest}')

