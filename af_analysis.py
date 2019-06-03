# Use viiflow to calculate a polar
import viiflow as vf
import numpy as np

def viiflowPolar(X,aoarange,RE=1e6,ncrit=9.0,Mach=0.0,GFheight = 0.0):
    # Settings
    s = vf.setup(Re=RE,Ma=Mach,ncrit=ncrit,alpha=aoarange[0])
    # Internal iterations
    s.itermax = 100
    s.silent=True
    # I will put the results here
    alv = []
    clv = []
    cdv = []
    cmv = []
    N = X.shape[1]
    index_gf = N-1
    vd = 0*X[0,:]
    vd[index_gf] = GFheight
	


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
                x[p.foils[0].N::p.foils[0].N+p.wakes[0].N-1]+=vd[-1]
                init = False
                if GFheight>0:
                    xtrans = p.foils[0].X[0,index_gf]-0.02
                    vf.set_forced_transition(bl,p,[],[xtrans])

            # Run viiflow
            [x,flag,_,_,_] = vf.iter(x,bl,p,s,None,None,vd)

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
                ue_gf = bl[0].bl_fl.nodes[index_gf].ue
                cdv.insert(pos,bl[0].CD+GFheight*(0.5*ue_gf**2))
                
            else:
                failed += 1
                if failed>=3:
                    print("Exiting after three failed AOA")
                    break
                init = True
    
    return (alv,clv,cdv,cmv,bl,p)