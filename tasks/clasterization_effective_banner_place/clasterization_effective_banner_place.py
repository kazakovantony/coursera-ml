import numpy as np
import pandas as pd
import math
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

"""
csv = []
delimiter_to_comma = '|'

with open("checkins.dat", "r") as f:
    for line in f:

        if line[0] == '-':
            continue

        converted_line = ''
        for c in line:
            if c == '' or c == ' ':
                continue
            if c == delimiter_to_comma:
                converted_line += ','
            else:
                converted_line += c

        converted_line = converted_line.strip()
        converted_line = converted_line.lstrip()
        csv.append(converted_line)


print(csv)

with open('csv.dat', 'w') as f:
    for item in csv:
        f.write("%s\n" % item)
"""

data = pd.read_csv('csv.dat')
data['latitude'].replace('', np.nan, inplace=True)
data.dropna(subset=['latitude'], inplace=True)
rows_to_process = 100000
latitude_ind = 3
longitude_ind = 4
data_to_work = data.head(rows_to_process)[[latitude_ind, longitude_ind]]

X = data_to_work.as_matrix(columns=data_to_work.columns)
bandwidth = 0.4
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
ms.cluster_centers_

occurrence_per_cluster = {}
for label in ms.labels_:
    if label in occurrence_per_cluster:
        occurrence_per_cluster[label] += 1
    else:
        occurrence_per_cluster[label] = 1

for e in ms.cluster_centers_:
    if 0.008 > e[0] > 0.007:
        print(e)
    if 0.008 > e[1] > 0.007:
        print(e)


def distance(cluster_center, office):
    return math.sqrt(((cluster_center[0] - office[0]) ** 2) + ((cluster_center[1] - office[1]) ** 2))


def sort_rule(e):
    return e[0]


offices = [
    [33.751277, -118.188740, 25.867736, -80.324116, 51.503016, -0.075479, 52.378894, 4.885084, 39.366487, 117.036146,
     -33.868457, 151.205134]]

result = []
current_cluster = 0
for e in ms.cluster_centers_:
    if occurrence_per_cluster[current_cluster] >= 15:
        for office in offices:
            result.append((distance(e, office), e))
    current_cluster += 1


result.sort(key=sort_rule)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k - 1
cluster_center = cluster_centers[k - 1]
plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
         markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
