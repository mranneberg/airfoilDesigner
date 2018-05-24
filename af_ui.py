import matplotlib.pyplot as plt
import numpy as np
def bezierPlot(ax,P):
    plt.plot([P[0,0],P[1,0]],[P[0,1],P[1,1]],"o-",color="#AAAAAA")
    for k in range(3,10,3):
        plt.plot([P[k-1,0],P[k,0],P[k+1,0]],[P[k-1,1],P[k,1],P[k+1,1]],"o-",color="#AAAAAA")
    plt.plot([P[11,0],P[12,0]],[P[11,1],P[12,1]],"o-b",color="#AAAAAA")

def savePoints(filename,X,name):
    np.savetxt(filename, X[:,::-1].T, fmt='  %1.8f  %1.8f', newline='\n', header=name,comments='')