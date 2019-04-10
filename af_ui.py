import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# Import some widgets
import ipywidgets as widgets
from ipywidgets import interactive, interact
from ipywidgets import FloatSlider

def bezierPlot(P):
    plt.plot([P[0,0],P[1,0]],[P[0,1],P[1,1]],"o-",color="#AAAAAA")
    for k in range(3,10,3):
        plt.plot([P[k-1,0],P[k,0],P[k+1,0]],[P[k-1,1],P[k,1],P[k+1,1]],"o-",color="#AAAAAA")
    plt.plot([P[11,0],P[12,0]],[P[11,1],P[12,1]],"o-b",color="#AAAAAA")

def designPlot(ax,handles,P,X):


    if len(handles)==0:
        handles.append(ax.plot(X[0,:],X[1,:],"+",color="#AAAAAA")[0])
        handles.append(ax.plot(X[0,:],X[1,:],"-k")[0])

        handles.append(ax.plot([P[0,0],P[1,0]],[P[0,1],P[1,1]],"o-",color="#AAAAAA")[0])
        handles.append(ax.plot([P[11,0],P[12,0]],[P[11,1],P[12,1]],"o-b",color="#AAAAAA")[0])
        for k in range(3,10,3):
            handles.append(ax.plot([P[k-1,0],P[k,0],P[k+1,0]],[P[k-1,1],P[k,1],P[k+1,1]],"o-",color="#AAAAAA")[0])

        plt.axis('equal')
    else:
        handles[0].set_xdata(X[0,:])
        handles[0].set_ydata(X[1,:])
        handles[1].set_xdata(X[0,:])
        handles[1].set_ydata(X[1,:])

        handles[2].set_xdata([P[0,0],P[1,0]])
        handles[2].set_ydata([P[0,1],P[1,1]])
        handles[3].set_xdata([P[11,0],P[12,0]])
        handles[3].set_ydata([P[11,1],P[12,1]])
        index = 3
        for k in range(3,10,3):
            index+=1
            handles[index].set_xdata([P[k-1,0],P[k,0],P[k+1,0]])
            handles[index].set_ydata([P[k-1,1],P[k,1],P[k+1,1]])

    return handles

def savePoints(filename,X,name):
    np.savetxt(filename, X.T, fmt='  %1.8f  %1.8f', newline='\n', header=name,comments='')


# These Parameters define the discretization
N=widgets.FloatLogSlider(value = 200, base = 10,min=2, max=2.8, step=0.01, continuous_update=False,description="Nodes")
LEFAC=FloatSlider(value = 6, min=0, max=10, step=0.05, continuous_update=False,description="LE Weight")
TEFAC=FloatSlider(value = 2, min=0, max=10, step=0.05, continuous_update=False,description="TE Weight")
KAPFAC=FloatSlider(value = 3, min=0, max=10, step=0.05, continuous_update=False,description="Crv Weight")
REFTOPX0=FloatSlider(value = 1, min=0, max=1, step=0.01, continuous_update=False,description="Top Left")
REFTOPX1=FloatSlider(value = 1, min=0, max=1, step=0.01, continuous_update=False,description="Top Right")
REFBOTX0=FloatSlider(value = 1, min=0, max=1, step=0.01, continuous_update=False,description="Bot Left")
REFBOTX1=FloatSlider(value = 1, min=0, max=1, step=0.01, continuous_update=False,description="Bot Right")
REFVAL=FloatSlider(value = 1, min=1, max=5, step=0.01, continuous_update=False,description="Ref Weight")

from af_bezier import * 
#from af_ui import * 
from af_pac import * 
from af_analysis import * 