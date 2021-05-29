#!/usr/bin/env python
# coding: utf-8

# In[1]:


def makeUniqueUgVector(Ug,beam1=1,beam2=8,phase1=0,phase2=0,file=None):
    import numpy as np
    import numpy.linalg as la
    
    filepath = "/Users/chenjie/pyQEDA/pyQEDA_simulations/" + file + "/" + file + "_experimental_data" +'/'
    beamfilename = '_BeamList.txt'
    beamfile = filepath + beamfilename
    beams = np.asarray(np.loadtxt(beamfile,skiprows=1, usecols= range(1, 4)  ), dtype=int )
    
    '''The function makeUniqueUgVector requires the following information:
    * Ug:     structure factor vector (1D array of complex numbers)
    * beams:  Miller indices of the reflections corresponding to the Ug-vector
    * beam1:  index to the first reference beam in the list of beams
    * beam2:  index to the second reference beam in the list of beams
    * phase1: Phase that the 1st beam shall be set to (default = 0)
    * phase2: Phase that the 2nd beam shall be set to (default = 0)

    The function makeUniqueUgVector returns:
    * UgUnique: Structure factor vector with shifted unit cell origin, such that the two reference 
                phases are set to the fixed values
    * r0:       shift that has been applied to the unit cell origin (in fractional coordinates)'''
    
    # Compute r0 by solving above linear equation
    G = 2.0*np.pi*np.array([beams[beam1,0:2], beams[beam2,0:2]])
    dPhi1 = np.angle(np.exp(1j*phase1)/Ug[beam1])
    dPhi2 = np.angle(np.exp(1j*phase2)/Ug[beam2])
    dPhi = np.array([dPhi1,dPhi2])
    try:
        r0 = np.linalg.solve(G, dPhi)
    except:
        r0 = np.zeros(2)
        UgUnique = Ug
        print("The two beams seem to be colinear - please chose different beams.")

    # Now, let's apply this offset to the unit cell origin to all structure factors in the list. 
    Nbeams = beams.shape[0]
    UgUnique = np.zeros(Nbeams,dtype=complex)
    for j in range(Nbeams):
        dPhi1 = 2*np.pi*(np.dot(beams[j,0:2],r0))
        UgUnique[j] = Ug[j]*np.exp(1j*dPhi1)
        
    print("Applied shift of unit cell origin (fractional coordinates):",r0[0],r0[1])
    for j in range(beams.shape[0]):
        print("Ug[",j,"]:",np.round(np.abs(Ug[j]),5),np.round(np.angle(Ug[j]),5),
                          " => ",np.round(np.abs(UgUnique[j]),5),np.round(np.angle(UgUnique[j]),5))
    
    return UgUnique, r0


# In[ ]:




