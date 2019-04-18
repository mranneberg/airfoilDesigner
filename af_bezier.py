'''
## Bezier discretization
For efficient discretization by arc-length we need the arclength along a 4-point cubic bezier curve.
The 4 point bezier curve is given by
$$ (1-t)^3 p_0 + 3 t (1-t)^2 p_1 + 3 t^2 (1-t) p_2 + t^3 p_3 $$
The analytic closed form of arclength $s$ of a *quadratic* bezier curve is [1]
$$ s = (t+\frac{b}{a})\sqrt{c+2bt+at^2} + \frac{ac-b^2}{a^{3/2}}\arcsin{\frac{at+b}{\sqrt{ac-b^2}}} $$
with $c = \Delta x_1^2+\Delta y_1^2$, $b = \Delta x_1 (\Delta x_2 - \Delta x_1)$, $a = (\Delta x_2 - \Delta x_1)^2+(\Delta y_2 - \Delta y_1)^2$ where $\Delta x_i = x_{i+1}-x_{i}$

If one wants to draw a quadratic bezier given by $P_0, P_{12}, P_3$ with a cubic one, one would choose the additional nodes[2]
$$ P_1 = P_0 + \frac{2}{3} (P_{12}-P_0)$$
$$ P_2 = P_3 + \frac{2}{3} (P_{12}-P_3)$$
So why, for an approximation given $P_1,P_2$, not use
$$ P_{12} = \frac{3}{4}(P_1-P_0+P_2-P_3)+(P_0+P_3)/2$$

We need an efficient discretization from an arclength perspective. 
To that end the cubic bezier curve is divided into 4 quadratic approximations.
For these, the arclength is given by an analytic expression as above.

After knowing the arclength, we can discretize the arclength based in the curvature.
The above curvature calculation is based on [3]

Note, the above comments are irrelevant, arclength is now numerically evaluated. Just wasn't worth it.

'''


import numpy as np
from scipy import interpolate as intp # Would be cool to get rid of
import numba as nb
from numba import jit,njit, f8,i8


@njit(nb.types.Tuple((f8[:,:],f8[:,:]))(f8[:,:],f8))
def deCastDivide(P0,t0):
    # Divides a 4Point bezier into two 4 point beziers
    # Explicit deCasteljau
    P11 = (1-t0)*P0[0,:] + t0*P0[1,:]
    P12 = (1-t0)*P0[1,:] + t0*P0[2,:]
    P13 = (1-t0)*P0[2,:] + t0*P0[3,:]
    P21 = (1-t0)*P11 + t0*P12
    P22 = (1-t0)*P12 + t0*P13
    P31 = (1-t0)*P21 + t0*P22

    P1 = P0.copy()
    P2 = P0.copy()
    P1[0] = P0[0,:]
    P1[1] = P11
    P1[2] = P21
    P1[3] = P31 

    P2[0] = P31
    P2[1] = P22
    P2[2] = P13
    P2[3] = P0[3,:]
    
    return (P1,P2)

@njit(nb.types.Tuple((f8[:,:],f8[:,:],f8[:,:],f8[:,:]))(f8[:,:]))
def bezierQuart(P):
    (PP1,PP2) = deCastDivide(P,0.5)
    (PP11,PP12) = deCastDivide(PP1,0.5)
    (PP21,PP22) = deCastDivide(PP2,0.5)
    return (PP11,PP12,PP21,PP22)


def getLength(P):
    S = 0
    dSarr = np.zeros(2*16)
    PP = []
    k=0
    for i in range(4):
        PP.append(bezierQuart(P[i*3:i*3+4,:]))
        
    for i in range(4):
        for j in range(len(PP[i])):
            dS = arcLenNumeric(PP[i][j][0],PP[i][j][1],PP[i][j][2],PP[i][j][3],1)
            dSarr[k]=dS
            k+=1
    S = np.sum(dSarr)
    return (S,dSarr,PP)

def pointsByArcLength(P,SU):
    # SU is 0->1 and is the length from bottom to top because i follow Melin.
    # Until I change that.
    points = np.zeros((2,len(SU)))
    kappa = np.zeros(len(SU))
    t = np.zeros(len(SU))
    (S,dSarr,PPF) = getLength(P)
    SS = SU*S # non-normalized length
    sub = len(PPF[0])
    S0 = 0
    i = 0
    ip = -1
    j = 0
    jp = -1
    ts = 0
    k = 0
    for s in SS:
        
        # Check if next section
        while s>np.sum(dSarr[0:(i+1)*sub]) and i<=3:
            i+=1
            j = 0
            ts = 0
        
        # Over max
        if i>3:
            ts = 1
            continue
        
        # Check if new subsection
        while s>np.sum(dSarr[0:i*sub+j+1]) and j<=sub-1:
            j+=1
            ts = 0
            
        # Over max
        if j>sub-1:
            ts = 1
            continue
            
        # Recalc if new section
        if not ip==i:
            PP = PPF[i]
            
        # Get arclength
        S0 = np.sum(dSarr[0:(i*sub+j)])
        
        # Arclen by numeric integration
        ts = findArcLenNumeric(PP[j][0],PP[j][1],PP[j][2],PP[j][3],s-S0,ts)
        ts = min(ts,1.0)
        p = bezier4(PP[j][0],PP[j][1],PP[j][2],PP[j][3],np.zeros(1)+ts)
        if k>0:
            tprev = t[k-1]
        else:
            tprev = 0.0#np.array(0.0)
        kappa[k] = maxBezierCurvature(PP[j][0],PP[j][1],PP[j][2],PP[j][3],tprev,ts)
        
        points[0,k] = p[0]
        points[1,k] = p[1]
        ip = i
        jp = j
        t[k] = ts
        k+=1
    points[0,0] = P[0,0]
    points[1,0] = P[0,1]
    points[0,-1] = P[-1,0]
    points[1,-1] = P[-1,1]
    
    return (t,points,kappa)

@njit(f8[:,:](f8[:],f8[:],f8[:],f8[:],f8[:]))
def bezier4( P0, P1, P2, P3, t):
    x = np.zeros((2,t.size))
    for k in range(2):
        x[k,:] = (1-t)**3*P0[k]+3*t*(1-t)**2*P1[k]+3*t**2*(1-t)*P2[k]+t**3*P3[k]
    return x

@njit(f8[:,:](f8[:],f8[:],f8[:],f8[:],f8[:]))
def bezier4dt( P0, P1, P2, P3, t):
    x = np.zeros((2,t.size))
    A0 = P1-P0
    A1 = P2-P1
    A2 = P3-P2
    D0 = A1-A0
    D1 = A2-A1
    E0 = D1-D0
    for k in range(2):
        x[k,:] = 3*(A0[k]+2*t*D0[k]+t**2*E0[k])
    return x

@njit(f8(f8[:],f8[:],f8[:],f8[:],f8))
def arcLenNumeric(P0,P1,P2,P3,t):
    # Bezier Differential
    
    # Simpson-Rule:
    ts = np.arange(0,1.5,.5)*t
    BDT = bezier4dt( P0, P1, P2, P3, ts)
    for k in range(3):
        BDT[0,k] = np.sqrt(BDT[0,k]**2+BDT[1,k]**2)
    s = 1/6*(BDT[0,0]+4*BDT[0,1]+BDT[0,2])*t
    return s

@njit(f8(f8[:],f8[:],f8[:],f8[:],f8,f8))
def findArcLenNumeric(P0,P1,P2,P3,S,t0):
    t2 = t0
    for k in range(50):
        s = arcLenNumeric(P0,P1,P2,P3,t2)
        dsdt2 = (arcLenNumeric(P0,P1,P2,P3,t2+1e-4)-arcLenNumeric(P0,P1,P2,P3,t2-1e-4))/2e-4
        if abs(s-S)<1e-10:
            break
        
        if abs(dsdt2)<1e-5:
            dsdt2 = 1
        dt2 = -(s-S)/dsdt2
        lam = min(1,min(0.1,abs((s-S)*10))/abs(dt2))
        t2 += dt2

    return t2


@njit(f8(f8[:],f8[:],f8[:],f8[:],f8))
def bezierCurvature(P0,P1,P2,P3,t):
    A0 = P1-P0
    A1 = P2-P1
    A2 = P3-P2
    D0 = A1-A0
    D1 = A2-A1
    E0 = D1-D0
    dCdt = np.zeros(2,dtype=np.float64)
    #dCdt_dt = np.zeros(2)
    d2Cd2t = np.zeros(2,dtype=np.float64)
    #d2Cd2t_dt = np.zeros(2)

    # k(t) = |C'xC''|/|C'|
    for k in range(2):
        dCdt[k] = 3*(A0[k]+2*t*D0[k]+t**2*E0[k])
        #dCdt_dt[k] = 3*(2*D0[k]+2*t*E0[k])
        d2Cd2t[k] = 6*(D0[k]+t*E0[k])
        #d2Cd2t_dt[k] = 6*(E0[k])
    # |C'|
    abs_dCdt = np.sqrt(dCdt[0]**2+dCdt[1]**2)
    #abs_dCdt_dt = 1/2/abs_dCdt*(2*dCdt_dt[0]+2*dCdt_dt[1])
    # |C'xC''|
    abs_cross = abs(dCdt[0]*d2Cd2t[1]-dCdt[1]*d2Cd2t[0])
    #abs_cross_dt = np.sign(dCdt[0]*d2Cd2t[1]-dCdt[1]*d2Cd2t[0])*(dCdt_dt[0]*d2Cd2t[1]-dCdt_dt[1]*d2Cd2t[0] + dCdt[0]*d2Cd2t_dt[1]-dCdt[1]*d2Cd2t:dt[0])
    kap = 0
    #kap_dt = 0
    if abs_dCdt>0:
        kap = abs_cross/abs_dCdt**3
        #kap_dt = abs_cross_dt/abs_dCdt**3 - 3*abs_cross_dt/abs_dCdt**4*abs_dCdt_dt
    return kap

@njit(f8(f8[:],f8[:],f8[:],f8[:],f8,f8))
def maxBezierCurvature(P0,P1,P2,P3,t0,t1):
    # Return maximum curvature between t0 and t1
    #S ample three points, fit quadratic, and assume thats cool.
    dt = t1-t0
    if dt==0:
        return bezierCurvature(P0,P1,P2,P3,t0)
    t = np.arange(t0,t1+dt/2,dt/2)
    
    # at^2+bt+c = k0,...
    A = np.zeros((3,3))
    b = np.zeros(3)
    kap = np.zeros(3)
    for k in range(3):
        kap[k] = bezierCurvature(P0,P1,P2,P3,t[k])
        A[k,:] = [t[k]**2, t[k], 1]
    coef = np.linalg.solve(A,kap)
    
    # Maximum of ax²+bx+c
    # zero-grad
    # 2*a*t = -b -> t = -b/2/a
    # However, if minimum...
    # 2*a < 0 !
    if coef[0]<0:
        tmax = -coef[1]/2/coef[0]
        return bezierCurvature(P0,P1,P2,P3,tmax)
    elif kap[2]>kap[0]:
        return kap[2]
    else:
        return kap[0]    
        

@njit(f8(f8[:],f8[:],f8[:]))
def localDiscreteCurvature(P,PM,PP):
    # Given a Point P and its neigbors PM an PP, calculate curvature
    # http://paulbourke.net/geometry/circlesphere/
    ma = (P[1]-PM[1])/(P[0]-PM[0])
    mb = (PP[1]-P[1])/(PP[0]-P[0])
    
    if mb==ma:
        return 0
    
    xc = (ma*mb*(PM[1]-PP[1])+mb*(PM[0]+P[0])-ma*(P[0]+PP[0]))/(2*(mb-ma))
    
    if abs(ma)<=1e-7:
        return 0
    yc = -1/ma*(xc-(PM[0]+P[0])/2) + (PM[1]+P[1])/2
    rc = np.sqrt((P[0]-xc)**2+(P[1]-yc)**2)
    
    if rc>0:
        kap = 1/rc
    else:
        kap = 0
        
    return kap

@njit(f8[:](f8[:,:]))
def discreteCurvature(Points):
    Curvature = np.zeros(Points.shape[1])
    for k in range(1,Curvature.shape[0]-1):
        Curvature[k] = localDiscreteCurvature(Points[:,k],Points[:,k-1],Points[:,k+1])
    Curvature[0] = Curvature[1]
    Curvature[-1] = Curvature[-2]
    return Curvature



'''
## How to discretize
Aims: 

* With more curvature, there should be more points per arclength
* At the trailing edge, the distance of the last two nodes should be identical on top and bottom side.
* A fine discretization at the leading edge (tbd, actually it would be nice to discretize the stagnation points)
* Just because there is no curvature and neither leading nor trailing edge, do not make huge steps.

For the designs above, that should be quite straightforward. The leading edge is at S(P[6]), The trailing edge at P[0] and P[12]. The curvature is given. Parameters: $N$, the number of nodes. That is, we need a vector ds of node increments with $\sum ds = S_f, ds>0, ds[0]=ds[-1]$ and this increment at its arclength position $s[i]=\sum_0^i ds[k]$ is propotional to $(\kappa(s[i]))^{-1}$. But the $ds$ should vary smoothly. Also, we want $ds[k]$ higher *near* the leading edge and near the trailing edge. To smooth these things, we start with a fine (i.e. 5 x $N$ or something like that), equidistant panelling of the airfoil and sample the curvature. We build a non-smooth function of demand-density by 
$$ d(s)=1+a_{\kappa}\frac{\kappa(s)}{\kappa_{max}}+a_{LE}\delta(s-LE<DS)+b_{TE}\delta(s-TE<DS) $$.
This density is then smoothed by a moving average of, let's say, $10 DS$.
Then, we know that with a yet unknown scaling factor $\gamma$
$$ ds[i] = \gamma d( s[i] )^{-1} $$
$$ s[i]=\sum_0^i ds[j] = \gamma \sum_0^i d(s[i])^{-1} $$
$$ s[N] = S_f $$
Ultimately, only the last equation is defining $\gamma$.
$$ \gamma \sum_0^N d(s[i])^{-1} = S_f$$
That is solved in a double Newton iteration:

Local problem for $s[k+1]$ while marching
$$ s[0] = 0 $$
$$ s[k+1] = s[k-1] + \gamma d(s[k+1])^{-1} $$

Global problem for $\gamma$ after marching
$$ s[N] = S_f$$
'''
#@jit(nb.types.Tuple((f8[:,:],f8[:]))(f8[:,:],i8,f8,f8,f8,f8[:],f8))
def repanelArclength(X,N,LEFAC,TEFAC,KAPFAC,REFLIMS,REFVAL):
    # The Curvature is given at panels S.

    Curvature = discreteCurvature(X)
    ND = len(Curvature)

    S = np.zeros(ND)
    for k in range(1,ND):
        S[k] = S[k-1]+np.sqrt((X[0,k]-X[0,k-1])**2+(X[1,k]-X[1,k-1])**2)

    # Rough estimate on ds
    dsest = S[-1]/N
    
    PC = .5

    DFUN = (1 + KAPFAC*(Curvature**PC)/np.mean(Curvature**PC) + TEFAC**2*(1-np.fmin(3*(1.0-X[0,:]),1.0))**2 + LEFAC**2*(1-np.fmin(3*X[0,:],1.0))**2  )**-1
    DFUNN = DFUN.copy()
    
    # This should be roughly 50 for 1000 points, or 5%
    WINDOW = int(0.05*ND)
    for k in range(ND):
        NQ = 3
        DFUNN[k] = DFUN[k]**NQ
        for j in range(0,WINDOW+1):
            kpj = (k+j)%ND
            kmj = (k-j)%ND # Incidentally, this leads to similar TE distances on top and bottom
            DFUNN[k] += (DFUN[kpj]**NQ+DFUN[kmj]**NQ)
        DFUNN[k] = (DFUNN[k])**(1.0/NQ)/(2*WINDOW+1)

    up = True
    minx = np.argmin(X[0,:])
    for k in range(ND):
        if up:
            if X[0,k]<=REFLIMS[1] and X[0,k]>=REFLIMS[0]:
                DFUNN[k]/=REFVAL
            if k>=minx:
                up = False
        else:
            if X[0,k]>=REFLIMS[2] and X[0,k]<=REFLIMS[3]:
                DFUNN[k]/=REFVAL
    
    DFUNN = DFUNN/np.mean(DFUNN)
    
    # We look for a scaling factor \gamma
    IT = intp.InterpolatedUnivariateSpline(S, DFUNN, k=3,ext=3)
    gamma = dsest/np.mean(DFUNN) # dS=gamma*DFUNN(f)->gamma~ds/gamma 
    
    # Once up.
    converged = 0
    SR = np.zeros(N)
    for it in range(200):
        SEND_GAM = 0
        for k in range(1,N):
            DS = IT(SR[k-1])
            DS_S = (IT(SR[k-1]+0.0000001)-DS)/0.0000001
            SR[k] = SR[k-1]+gamma*DS
            SEND_GAM += gamma*DS_S*SEND_GAM + DS
        
        if abs(SEND_GAM)<=0:
            converged+=1
            break
        dGam = -(SR[-1]-S[-1])/SEND_GAM
        
        if abs(dGam)<1e-6:
            converged+=1
            break
        lam = min(1,0.05/(abs(dGam)/abs(gamma)))
        gamma += lam*dGam
        if abs(SR[-1]-S[-1])<=1e-6:
            converged+=1
            break
    

    # Once down.
    SRUP = np.zeros(N)+S[-1]
    for it in range(100):
        SEND_GAM = 0
        for k in range(N-1,0,-1):
            DS = IT(SRUP[k])
            DS_S = (IT(SRUP[k]-0.0000001)-DS)/0.0000001
            SRUP[k-1] = SRUP[k]-gamma*DS
            SEND_GAM += gamma*DS_S*SEND_GAM - DS
        
        if abs(SEND_GAM)<=0:
            converged+=1
            break
        dGam = -(SRUP[0])/SEND_GAM
        
        if abs(dGam)<1e-6:
            converged+=1
            break
        lam = min(1,0.5/(abs(dGam)/abs(gamma)))
        gamma += lam*dGam
        if abs(SRUP[0])<=1e-6:
            converged+=1
            break
        
    # Mean of up and down.
    #if converged<2:
        #print("Not Converged"+str(converged))
    #    return -1
    
    SR = (SR+SRUP)*0.5

    # Interpolate ArcPoints to new
    XN = np.zeros((2,int(N)))
    s = intp.InterpolatedUnivariateSpline(S,X[0,:], k=3)
    XN[0,:] = s(SR)
    s = intp.InterpolatedUnivariateSpline(S,X[1,:], k=3)
    XN[1,:] = s(SR)
    
    return (XN,SR)
