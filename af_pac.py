import numpy as np

def B64read(strin):
    assert len(strin)==15
    str64 = """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"""
    k = np.zeros(15)
    for i in range(15):
        k[i] = str64.find(strin[i])/63.0
    return k

def B64write(k):
    assert len(k)==15
    str64 = """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"""
    str=''
    for i in range(15):
        str += str64[int(k[i]*63)]
    return str


def foilPoints(k):
    Hmax = .3/2 # Maximum thickness t/c
    Gmax = .05/2 # Maximal TE gap
    almax = 0.35 # Maximum TE angle [rad]
    betmax = 1 # Maximum boat angle [rad] 

    # Upper Side
    c = k[5-1]*Hmax
    y8 = k[1-1]*c
    PU = k[3-1]
    x10 = PU
    x11 = PU+k[4-1]*(1-PU)
    x9 = PU-k[2-1]*PU
    
    # Lower Side
    b = k[10-1]*Hmax
    y6 = -k[6-1]*b
    PL = k[8-1]
    x3 = PL+k[9-1]*(1-PL)
    x4 = PL
    x5 = PL-k[7-1]*(PL)
    
    # Trailing edge
    g = k[15-1]*Gmax
    a = g*0.5
    al = (2*k[14-1]-1)*almax
    bet = k[13-1]*betmax
    
    alup = al+bet*0.5
    allo = al-bet*0.5
    
    # First Limit: Not outside bounding box
    # c >= a+V*sin(al)
    Vmax = 1.0
    if alup>0 and c>a:
        Vmax = np.minimum(Vmax,(c-a)/np.sin(alup))
    if alup<0 and b>3*a:
        Vmax = np.minimum(Vmax,(b-3*a)/np.sin(-alup))
    # Second Limit: No further than x11
    # x11 <= 1-V*k*cos(al)
    # x11-1 <= -V*cos(al)
    # (1-x11)/cos(al) >= V
    Vmax = np.minimum(Vmax,(1-x11)/np.cos(alup))
    
    VUp = k[11-1]*Vmax
    x12 = 1-VUp*np.cos(alup)
    y12 = a+VUp*np.sin(alup)
    
    # First Limit: Not outside bounding box
    # c >= a+V*k=1*sin(al)
    Vmax = 1.0
    if allo<0 and b>a:
        Vmax = np.minimum(Vmax,(b-a)/np.sin(-allo))
    if allo>0 and c>3*a:
        Vmax = np.minimum(Vmax,(c-3*a)/np.sin(allo))
    # Second Limit: No further than x11
    # x11 <= 1-V*k*cos(al)
    # x11-1 <= -V*cos(al)
    # (1-x11)/cos(al) >= V
    Vmax = np.minimum(Vmax,(1-x3)/np.cos(allo))
    VLo = k[12-1]*Vmax
    x2 = 1-VLo*np.cos(allo)
    y2 = -a+VLo*np.sin(allo)
    
    
    #x2,-x3,-x4,-x5,-x9,-x10,-x11,x12,y2,-y6,-y8,y12,-a,-b,-c
    
    P = np.zeros((13,2))
    P[0,0] = 1
    P[0,1] = -a
    P[1,0] = x2
    P[1,1] = y2
    P[2,0] = x3
    P[2,1] = -b
    P[3,0] = x4
    P[3,1] = -b
    
    P[4,0] = x5
    P[4,1] = -b
    P[5,1] = y6
    P[6,0] = 0
    P[7,1] = y8
    P[8,0] = x9
    P[8,1] = c
    
    P[9,0] = x10
    P[9,1] = c
    P[10,0] = x11
    P[10,1] = c
    P[11,0] = x12
    P[11,1] = y12
    P[12,0] = 1
    P[12,1] = a
    
    return P