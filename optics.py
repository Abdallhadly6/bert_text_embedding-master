from sklearn.cluster import OPTICS
import numpy as np
data = []
with open('vectors2.txt') as f:
   content = f.readlines()
temp = [x.strip() for x in content]

for i in range(len(temp)):
    temp2 =( temp[i].split())
    data.append(temp2)
vectors = [[float(v) for v in r[0].split(',')] for r in data]

arr = np.array(vectors)

clustering = OPTICS(min_samples=2).fit(arr)
print((clustering.labels_[0]))
print(len(clustering.labels_))


