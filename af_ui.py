import matplotlib.pyplot as plt
import numpy as np
# Import some widgets
import ipywidgets as widgets
from ipywidgets import interactive, interact
from ipywidgets import FloatSlider


def bezierPlot(ax,P):
    plt.plot([P[0,0],P[1,0]],[P[0,1],P[1,1]],"o-",color="#AAAAAA")
    for k in range(3,10,3):
        plt.plot([P[k-1,0],P[k,0],P[k+1,0]],[P[k-1,1],P[k,1],P[k+1,1]],"o-",color="#AAAAAA")
    plt.plot([P[11,0],P[12,0]],[P[11,1],P[12,1]],"o-b",color="#AAAAAA")

def savePoints(filename,X,name):
    np.savetxt(filename, X[:,::-1].T, fmt='  %1.8f  %1.8f', newline='\n', header=name,comments='')


# These Parameters define the discretization
N=widgets.FloatLogSlider(value = 200, base = 10,min=2, max=2.8, step=0.01, continuous_update=False,description="Nodes")
LEFAC=FloatSlider(value = 6, min=0, max=10, step=0.05, continuous_update=False,description="LE Weight")
TEFAC=FloatSlider(value = 2, min=0, max=10, step=0.05, continuous_update=False,description="TE Weight")
KAPFAC=FloatSlider(value = 3, min=0, max=10, step=0.05, continuous_update=False,description="Crv Weight")