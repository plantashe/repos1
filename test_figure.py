import os
from matplotlib import axes
import torch
import logging
import numpy as np
from scipy import spatial, interpolate
#from utils.plot_utils import mesh_plotter_2d
#from utils.plot_utils import mesh_plotter_2d_notrii
from utils.models import model_saver
import matplotlib.pyplot as plt

'''
def mesh_plotter_2d(coords, simplices, step=None, ex_path="./mesh_data", name="mesh"):
    """
    function to plot triangular meshes
    """
    assert coords.shape[1] == 2
    plt.figure(figsize=(20, 20), dpi=100)    
    plt.triplot(coords[:, 0], coords[:, 1], simplices)

def mesh_all(coords):
    plt.scatter(coords[:, 0], coords[:, 1],c='b')
    


tspace = np.linspace(0, 1, 3 )
xspace = np.linspace(0, 1, 3 )
T, X = np.meshgrid(tspace, xspace)
Xgrid = np.vstack([T.flatten(),X.flatten()]).T
print(Xgrid)
seed_points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
tri = spatial.Delaunay(seed_points)
mesh_plotter_2d(Xgrid, tri.simplices, 1)
print(tri.simplices)
#mesh_all(Xgrid)
plt.show()

points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
'''

'''
tspace = np.linspace(0, 1, 3 )
xspace = np.linspace(0, 1, 3 )
T, X = np.meshgrid(tspace, xspace)
Xgrid = np.vstack([T.flatten(),X.flatten()]).T
plt.scatter(Xgrid[:, 0], Xgrid[:, 1],c='b')
plt.scatter(0.5,0.5,c='r')

points = np.array([[0, 0], [0.5, 1.], [1, 0.5],[0.5,0.3],[0.2,0.6],[0.65,0.3],[0.6,0.65]])
from scipy.spatial import Delaunay
tri = Delaunay(points)
points_ex=np.array([[0.5,0.3],[0.2,0.6],[0.65,0.3],[0.6,0.65]])


plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.plot(points_ex[:,0], points_ex[:,1], 'o',c='yellow')
plt.xticks([])
plt.yticks([])
plt.show()
'''

tspace = np.linspace(0, 1, 3 )
xspace = np.linspace(0, 1, 3 )
T, X = np.meshgrid(tspace, xspace)
Xgrid = np.vstack([T.flatten(),X.flatten()]).T
fig,axes=plt.subplots(1,2,figsize=(10,10))
axes[1].scatter(Xgrid[:, 0], Xgrid[:, 1],marker='o',s=120,c='dimgray',label="collection points")
axes[1].scatter(0.5,0.5,marker='o',s=120,c='green',label="goal points")
points = np.array([[0, 0], [0.5, 1.], [1, 0.5],[0.5,0.3],[0.2,0.6],[0.65,0.3],[0.6,0.65]])
points_base=np.array([[0, 0], [0.5, 1.], [1, 0.5]])
from scipy.spatial import Delaunay
tri = Delaunay(points)
points_ex=np.array([[0.5,0.3],[0.2,0.6],[0.65,0.3],[0.6,0.65]])
axes[1].triplot(points[:,0], points[:,1], tri.simplices)
axes[1].scatter(points_ex[:,0], points_ex[:,1],marker='o',s=120,c='r',label="OIP seeds")
axes[1].scatter(points_base[:,0], points_base[:,1],marker='o',s=120,c='b',label="interpolation seeds")
axes[1].xaxis.set_visible(False)
axes[1].yaxis.set_visible(False)
l_b=axes[1].legend(loc='best')
axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,prop = {'size':16}) 
l_b.get_frame().set_alpha(0.3)
axes[1].set_title("(b)",loc='center',y=-0.08,fontsize=20)


points = np.array([[0, 0], [0.5, 1.], [1, 0.5]])
points_base=np.array([[0, 0], [0.5, 1.], [1, 0.5]])
tri = Delaunay(points)
tri = Delaunay(points)
axes[0].triplot(points[:,0], points[:,1], tri.simplices)
axes[0].scatter(Xgrid[:, 0], Xgrid[:, 1],marker='o',s=120,c='dimgray',label="collection points")
axes[0].scatter(0.5,0.5,marker='o',s=120,c='green',label="goal points")
points_ex=np.array([[0.5,0.3],[0.2,0.6],[0.65,0.3],[0.6,0.65]])
axes[0].scatter(points_base[:,0], points_base[:,1], c='b',marker='o',s=120,label="interpolation seeds")
axes[0].xaxis.set_visible(False)
axes[0].yaxis.set_visible(False)
axes[0].set_title("(a)",loc='center',y=-0.08,fontsize=20)
plt.show()