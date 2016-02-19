# starter.py

import matplotlib.pyplot as plt
import scipy as sp

data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")

x = data[:,0]
y = data[:,1]

# plot the x y points with dots of size 10
plt.scatter(x, y, s=10)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],
           ['week %i' % w for w in range(10)])
plt.autoscale(tight=True)
# draw a slightly opaque, dashed grid
plt.grid(True, linestyle='-', color='0.75')
plt.show()


