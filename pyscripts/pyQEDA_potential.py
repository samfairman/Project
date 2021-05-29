#Sam Fairman March 2021 - adapted from Matlab script by C.T. Koch

def displayPotentialFromUg(folder=None, potSize=([256,256]), Ncells=([1,1]),showImag=False,doFlip=True ,thresh = 0.015, beta = 0.9, InputUg=None, InputBeams=None, returnComplexUg=True, scaling=1): 
	
	import os
	import pyQEDA as pq #note this python source file/jupyter notebook should not be called 'pyQEDA.py' as this is reserved for the pyQEDA module as will cause a conflict if named so
	import numpy as np
	import math
	import matplotlib.pyplot as plt
	from copy import deepcopy
	currentPath = os.getcwd()
	#currentPath = "/Users/chenjie/pyQEDA"


	def displayPotential_2D(UgList,beams,potSize=([256,256]),Ncells=([1,1]),showImag=True,doFlip=False, thresh=thresh, beta = beta):
		# This function displays the real-space potential corrsponding to a set of
		# structure factors.  
		# Input: Ncells = number of unit cells
		# potSize: size of potential map in pixels
		NxMid = np.int(np.floor(potSize[1]/2)+1)
		NyMid = np.int(np.floor(potSize[0]/2)+1)
		potMap = np.zeros(potSize,dtype=complex)
		#thresh = 0.015
		#beta = 0.9


		for j in range(len(UgList)):
			potMap[NyMid+beams[j,1]*Ncells[1],NxMid+beams[j,0]*Ncells[0]] = UgList[j]

		potMap = np.fft.ifft2(np.fft.ifftshift(potMap))

		if showImag:
			plt.imshow(potMap.real)
			plt.title(f"Potential Map Real - {folder}\n\n")
			plt.colorbar()
			plt.show()
			plt.imshow(potMap.imag)
			plt.title(f"Potential Map Imaginary - {folder}\n\n")
			plt.colorbar()
			plt.show()
			plt.imshow(np.abs(potMap))
			plt.title(f"Potential Map Abs - {folder}\n\n")
			plt.colorbar()
			plt.show()

		# Do charge flipping
		if doFlip:        
			tthresh = thresh*np.max(potMap)
			ind = np.asarray(np.where(potMap < -tthresh) )
			potMap[ np.nonzero(potMap < -tthresh) ] =  - tthresh - beta * ( potMap[ potMap < -tthresh ]  + tthresh )

			if showImag:
				plt.imshow(potMap.real)
				plt.title(f"Potential Map Charge Flipped - Real - {folder}\n\n")
				plt.colorbar()
				plt.show()
				plt.imshow(potMap.imag)
				plt.title(f"Potential Map Charge Flipped - Imaginary - {folder}\n\n")
				plt.colorbar()
				plt.show()
				plt.imshow(np.abs(potMap))
				plt.title(f"Potential Map Charge Flipped Abs - {folder}\n\n")
				plt.colorbar()
				plt.show()

			pmf = np.fft.fftshift(np.fft.fft2(potMap))			
			
			for j in range(len(UgList)):
				UgList[j] = pmf[NyMid+beams[j,1]*Ncells[1],NxMid+beams[j,0]*Ncells[0]]
			
		
		return UgList, potMap



	def displayPotentialFromUgFile(filepath=folder,potSize=potSize,Ncells=Ncells,showImag=showImag,doFlip=doFlip, thresh=thresh, beta=beta):
		beamfilename = '_BeamList.txt'
		beamfile = filepath + beamfilename
		UgReal = np.loadtxt(beamfile,skiprows=1, usecols= range(4, 5)  )
		UgImag = np.loadtxt(beamfile,skiprows=1, usecols= range(5, 6)  )
		Ug = np.array(range(UgReal.size), dtype=complex)
		Ug.real = UgReal
		Ug.imag = UgImag
		beams = np.asarray(np.loadtxt(beamfile,skiprows=1, usecols= range(1, 4)  ), dtype=int ) 
		Ugcopy = deepcopy(Ug)	
		UgList, potMap = displayPotential_2D(Ug,beams,potSize=potSize,Ncells=Ncells,showImag=showImag,doFlip=doFlip, thresh=thresh, beta=beta)		
		print(f'\nDifference pre and post Ug charge flipping\n{Ugcopy[0]} -> {UgList[0]} ')

		
		return UgList, potMap, beams

	def displayPotentialFromUgVar(InputUg=InputUg, InputBeams=InputBeams, potSize=potSize,Ncells=Ncells,showImag=showImag,doFlip=doFlip, thresh=thresh, beta=beta):
		Ugcopy = deepcopy(InputUg)	
		UgList, potMap = displayPotential_2D(InputUg,InputBeams,potSize=potSize,Ncells=Ncells,showImag=showImag,doFlip=doFlip, thresh=thresh, beta=beta)		
		if doFlip:
			print(f'\nDifference pre and post Ug charge flipping\n{Ugcopy[0]} -> {UgList[0]} ')
			print(f'\nMax Difference =\n{np.max(UgList-Ugcopy)}')
		return UgList, potMap

	
	#MAIN
	if folder!=None:
		filepath = "/Users/chenjie/pyQEDA/pyQEDA_simulations/" + folder + "/" + folder + "_experimental_data" +'/'
		[UgList, potMap, beams] = displayPotentialFromUgFile(filepath)
		if returnComplexUg:
			return UgList, potMap
		else:
			return [(np.ravel([UgList.real,UgList.imag],'F')), potMap, beams]
	else:
		print(f'\n\n\n\n{InputUg}')
		UgComplex = np.zeros(np.int(InputUg.size/2),dtype=complex)
		UgComplex.real = InputUg[::2] 
		UgComplex.imag = InputUg[1::2] 
		[UgList, potMap] = displayPotentialFromUgVar(UgComplex / scaling ,InputBeams)
		
		if returnComplexUg:
			return [UgList * scaling, potMap]
		else:
			return (np.ravel([UgList.real,UgList.imag] * scaling ,'F')), potMap, beams

	
