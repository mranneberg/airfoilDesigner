# Use viiflow to calculate a polar
import viiflow as vf
import numpy as np

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
                (p,bl,x) = vf.init([X[:,::-1]],s)
                init = False
            res = None
            grad = None

            # Run viiflow
            [x,flag,res,grad,_] = vf.iter(x,bl,p,s,res,grad)

            # If converged add to cl/cd vectors (could check flag as well, but this allows custom tolerance to use the results anyways)
            if np.sqrt(np.dot(res.T,res))<1e-3:
                failed = 0
                if uplo==0:
                    alv.append(alpha)
                    clv.append(p.CL)
                    cmv.append(p.CM)
                    cdv.append(bl[0].CD)
                    #print('AL: %f CL: %f CD: %f' % (alpha,clv[-1],cdv[-1]))
                else:
                    alv.insert(0,alpha)
                    clv.insert(0,p.CL)
                    cmv.insert(0,p.CM)
                    cdv.insert(0,bl[0].CD)
                    #print('AL: %f CL: %f CD: %f' % (alpha,clv[0],cdv[0]))
                
            else:
                failed += 1
                if failed>=3:
                    print("Exiting after three failed AOA")
                    break
                init = True
    
    return (alv,clv,cdv,cmv,bl,p)