from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt

grid = np.loadtxt('d:\\Balt-P\\Grids\\grid2m.txt')
mask = np.loadtxt('d:\\Balt-P\\Grids\\mask2m.txt')
len_i = len(grid)/2
len_j = len(grid[0])

# plt.figure(figsize=(200, 200))
# resolution: c (crude), l (low), i (intermediate), h (high), f (full)

d_ind = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

#m = Basemap(projection='merc', llcrnrlat=53, urcrnrlat=66, llcrnrlon=11, urcrnrlon=32, lat_ts=20, resolution='i')
m = Basemap(projection='merc', llcrnrlat=59, urcrnrlat=61, llcrnrlon=23, urcrnrlon=30.5, lat_ts=60, resolution='h')
m.drawcoastlines()
m.fillcontinents(color='white', lake_color='#eeeeee')
m.drawparallels(np.arange(59, 62, 0.5), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(23, 31, 1.5), labels=[0, 0, 0, 1])
m.drawmapboundary(fill_color='#eeeeee')
m.drawrivers(linewidth=0.5)
for i in range(len_i):
    for j in range(len_j):
        if mask[i, j] != 0:
            coo = np.array([grid[i, j], grid[i + len_i, j]])
            if (coo[0] < 22.5) or (coo[0] > 31.5) or (coo[1] < 58.5) or (coo[1] > 62.5):
                continue
            print(i, j)
            for dk in range(4):
                di, dj = d_ind[dk]
                if (i + di >= 0) and (i + di < len_i) and (j + dj >= 0) and (j + dj < len_j):
                    m_1 = 0.5 * (coo[0] + grid[i + di, j + dj])
                    m_2 = 0.5 * (coo[1] + grid[i + di + len_i, j + dj])
                    geo_nb = m(np.array([coo[0], m_1]), np.array([coo[1], m_2]))
                    if mask[i, j] + mask[i + di, j + dj] == 3:
                        m.plot(geo_nb[0], geo_nb[1], color='r', linewidth=5)
                    else:
                        m.plot(geo_nb[0], geo_nb[1], color='k', linewidth=2)



#coo = m(points[:, 0].flatten(), points[:, 1].flatten())
#m.plot(coo[0], coo[1], marker='o')
plt.title("Baltic sea")
plt.show()

