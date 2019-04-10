# Use viiflow to calculate a polar
import viiflow as vf
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"

def viiflowPolar(X,aoarange,RE=1e6,ncrit=9.0,Mach=0.0):
    # Settings
    Mach = 0.0
    s = vf.setup(Re=RE,Ma=Mach,ncrit=ncrit,alpha=aoarange[0])
    # Internal iterations
    s.itermax = 100
    s.silent=True
    # I will put the results here
    alv = []
    clv = []
    cdv = []
    cmv = []

    # Set-up and initialize based on inviscid panel solution
    # This calculates panel operator

    # Go over AOA range
    for uplo in range(2):
        init = True
        failed = 0
        if uplo==0:
            aoar = aoarange[np.int(np.ceil(len(aoarange)*0.5))::]
        else:
            aoar = aoarange[np.int(np.floor(len(aoarange)*0.5))::-1]
        for alpha in aoar:

            # Set current alpha and set res/grad to None to tell viiflow that they are not valid
            s.alpha = alpha

            if init:
                (p,bl,x) = vf.init(X,s)
                init = False

            # Run viiflow
            [x,flag,_,_,_] = vf.iter(x,bl,p,s)

            # If converged add to cl/cd vectors
            if flag:
                failed = 0
                if uplo==0:
                    pos = len(alv)
                else:
                    pos = 0
                alv.insert(pos,alpha)
                clv.insert(pos,p.CL)
                cmv.insert(pos,p.CM)
                cdv.insert(pos,bl[0].CD)
                
            else:
                failed += 1
                if failed>=3:
                    print("Exiting after three failed AOA")
                    break
                init = True
    
    return (alv,clv,cdv,cmv,bl,p)