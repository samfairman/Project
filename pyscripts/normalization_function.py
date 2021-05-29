#!/usr/bin/env python
# coding: utf-8

# In[2]:


def normalization(file = None,  origin_Ug=None):
    
    import numpy as np
    
    filepath = "/Users/chenjie/pyQEDA/pyQEDA_simulations/" + file + "/" + file + "_experimental_data" +'/'
    beamfilename = '_BeamList.txt'
    beamfile = filepath + beamfilename
    beams = np.asarray(np.loadtxt(beamfile,skiprows=1, usecols= range(1, 4)  ), dtype=int )

    Ug_normalized = []
    for i in range(len(beams)):
        g = i
        for j in range(len(beams)):
            if (beams[i] + beams[j]==0).all():
                minus_g = j
                        
        U_g = origin_Ug[g]
        U_minus_g = origin_Ug[minus_g]
        temp = 0.5*(U_g + np.conjugate(U_minus_g))
        U_g = temp
        U_minus_g = np.conjugate(temp)
        Ug_normalized.append(U_g)
    return Ug_normalized


# In[ ]:




