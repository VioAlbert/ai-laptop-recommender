import csv
import numpy as np

# variables

with open('data_normalized.csv') as csv_file:
  laptops = list(csv.reader(csv_file, delimiter=','))
  count_spec = len(laptops[0]) - 1
  for i in range(1, count_spec+1):
    for row in laptops:
      row[i] = float(row[i])
laptops = np.array(laptops, dtype='object')

with open('tests/input.csv') as csv_file:
  queries = list(csv.reader(csv_file, delimiter=','))
  queries.pop(0)
  for row in queries:
    row.pop(0)
    for i in range(0, count_spec):
      row[i] = float(row[i])

queries = np.array(queries, dtype='object')
count_queries = len(queries)
# print(queries)

with open('tests/output.csv') as csv_file:
  expected = list(csv.reader(csv_file, delimiter=','))
  expected.pop(0)
  for row in expected:
    for i in range(1, count_queries+1):
      row[i] = int(row[i])

expected = np.array(expected, dtype='object')
# print(expected)

dist_normalizer = np.sqrt(7)
sim_normalizer = 2

# adjustable variables

# threshold = 0.1933
# c_dist = 2
# c_sim = 5

threshold = 0.30
c_dist = 3
c_sim = 2

def get_nearest(query):
  # Find distance using euclidean distance and cosine distance

  nearest = np.empty((0, 2), dtype='object')

  for row in laptops:
    passed = True
    failed = True
    for i in range(1, count_spec+1):
      if row[i] < query[i-1]:
        passed = False
      if row[i] >= query[i-1]:
        failed = False

    # dist = np.linalg.norm(query - row[1:8]) / dist_normalizer

    dist = 0
    for i in range(1, count_spec+1):
      if row[i] < query[i-1]:
        dist += np.square(query[i-1] - row[i])
    dist = np.sqrt(dist)

    sim = (1 - (query.dot(row[1:8]) / (np.linalg.norm(query) * np.linalg.norm(row[1:8])))) / sim_normalizer
    val = (c_dist * dist + c_sim * sim) / (c_dist + c_sim)

    if passed == True:
      val += -100
    if failed == True:
      val += 100

    nearest = np.append(nearest, np.array([[row[0], val]], dtype='object'), axis=0)

  nearest = nearest[np.argsort(nearest[:,1])]

  return nearest

def eval(positive, negative, expected):
  true_positive = 0 # keluar bener, dan bener
  false_positive = 0 # keluar bener, tapi harusnya salah
  true_negative = 0 # keluar salah, dan salah
  false_negative = 0 # keluar salah, tapi harusnya bener

  for x in positive:
    for y in expected:
      if x[0] == y[0]:
        if y[1] == 1: true_positive += 1
        else: false_positive += 1
        break

  for x in negative:
    for y in expected:
      if x[0] == y[0]:
        if y[1] == 0: true_negative += 1
        else: false_negative += 1
        break

  print(true_positive, false_positive, true_negative, false_negative)

  precision = true_positive / (true_positive + false_positive)
  recall = true_positive / (true_positive + false_negative)
  F1 = 2 * precision * recall / (precision + recall)
  accuracy = (true_positive + true_negative) / 100
  return precision, recall, F1, accuracy



# Main

avg_precision = 0
avg_recall = 0
avg_F1 = 0
avg_accuracy = 0

for i in range(count_queries):
  query = queries[i]
  nearest = get_nearest(query)
  # print(len(nearest))
  positive = nearest[nearest[:,1] <= threshold]
  negative = nearest[nearest[:,1] > threshold]
  precision, recall, F1, accuracy = eval(positive, negative, expected[:, [0,i+1]])
  print(f'precision = {precision}, recall = {recall}, F1 = {F1}, accuracy = {accuracy}')
  avg_precision += precision
  avg_recall += recall
  avg_F1 += F1
  avg_accuracy += accuracy

avg_precision /= count_queries
avg_recall /= count_queries
avg_F1 /= count_queries
avg_accuracy /= count_queries

print(f'\nAverage: precision = {avg_precision}, recall = {avg_recall}, F1 = {avg_F1}, accuracy = {accuracy}')
