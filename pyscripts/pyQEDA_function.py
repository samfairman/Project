def pyQEDA(filename=None, Generate_random_diffs = False, N_training_data = 1, UseInputUgMasterList = False,\
				InputUg = None, scaling = 1, ReturnOutputUg = False, mode ='LARBED_SIM'):
	# 1. Ar + Master list
	# 2. expm = e^(itAlambda) = S
	# 3. S^2 = Ig -> GP -> Ug
	# use weak beam approximation after it works

	# program: qeda: 'Quantitative Electron Diffraction Analysis' 

		# First we read/create in the experimental conditions.
		# We need: 
		# - the microscope parameters (high tension)
		# - the (indexed) reciprocal lattice
		# - the zone axis
		# - the array of kt-vectors (this could also be defined as equally spaced, with some maximum value)
		# 
		# There are 2 options for this code: 
		# a) compute a LARBED pattern (requires knowledge of crystal structure)
		# b) analyze an experimental data set (requires measured intensities and spot positions)
		# We should first complete a), because we can then use that as input to b)
		############################################/
	#%%
	import os
	import pyQEDA as pq #note this python source file/jupyter notebook should not be called 'pyQEDA.py' as this is reserved for the pyQEDA module as will cause a conflict if named so
	import numpy as np
	import requests
	import math
	import matplotlib.pyplot as plt
	currentPath = os.getcwd()
	import sys

	#filename = "SrTiO3_001_Th200A_T30mrad_r5_maxg_1_100kV.qed" #enter the QEDA parameter file here (including filetype)
	try:
		filename = currentPath + "/parameter_files/" + filename
	except:
		print('no file found - running with default parameters')
	cfgFolder="/crystal_files/" #these input and output folders are appended to the beginnng of the string read from the .qed file
	outputFolder="/pyQEDA_simulations/"
	#imageFolder="/pyQEDA_simulations/" #these input and output folders are appended to the beginnng of the string read from the .qed file
	#if using experimental intensities
	expImageFolder = "/experimental_data/"
	expImageFolder = currentPath + expImageFolder

	#%% Default parameters if no input file is available

	# Let's start with a few settings:
	computeTermMaps = 0
	############################################/
	# Here are a few default parameters which should be overwritten by an input data file:
	n = ([0.0,0.0,1.0]) #surface normal (surface slab normal points against the beam)
	zone = ([0.0,0.0,1.0]) #which zone axis (crystal orientation)
	gx = ([1.0,0.0,0.0])    #the g vector
	ktOffset = ([0.0,0.0])
	highTension = 60    #the electron bean energy in kV
	gCutoff = 0.2      # cut off reflections at 3 1/A
	sgCutoff    = 0.025 # select reflections that are closer than 0.2 1/A to the Ewald sphere.
	thickness   = 500.0 # thickness in A.
	scanComp    = 0.0 #scan compensation can be used to descan the beam to ensure there are no overlapping beams, (with full scan compensation there is no rocking curve, only integrated spots)
	detectorNx  = 91 #size of the detector
	detectorNy  = detectorNx
	tiltRange_mrad = 40 
	discRadius_pixel = 5  #the scan pattern is a disc, this parameter chooses how many scan points to fit in one radius and the total scan points is then calculated to fit inside the disc.
	q , approxOrder = 0, 0
	termThresh = 0.0001 
	targetUg = 0.01
	cfgName = 'SrTiO3.cfg'
	#Generate_random_diffs = False

	#UseInputUgMasterList = False

	if UseInputUgMasterList:
		Generate_random_diffs = False #dont use both at the same time

	if Generate_random_diffs: 
		RandomUgMasterList = []
		random_diffs = []
		UgReal = []
		UgImag = []
		OutputUgList = []
	else:
		Generate_random_diffs = False #dont use both at the same time
		diff_pattern = []
		N_training_data = 1



	def read_parameter_file(filename,training_data):    
		with open(filename, "r") as f:
			for line in f:
				if 'mode:' in line:
					mode = line[line.index(':')+1:-1].strip(' ')
					print(f"mode = \t\t{mode}") 
				if 'cfg file:' in line:
					cfgName = currentPath + cfgFolder + line[line.index(':')+1:-1].strip(' ')
					print(f"cfg file = \t\t{cfgName}") 
				if 'output folder:' in line:
					outputName = line[line.index(':')+1:-1].strip(' ')
					if Generate_random_diffs:
						outputPath = currentPath + outputFolder + "/" + outputName + "/" + outputName + "_training_data_" + str(training_data)
					else:
						outputPath = currentPath + outputFolder + "/" + outputName + "/" + outputName
					print(f"created output folder = \t\t{outputFolder}")
					if os.path.exists(outputPath):
						print("file exists")
					else:
						pass
						os.makedirs( outputPath )
				if 'output file:' in line:
					imageName = line[line.index(':')+1:-1].strip(' ') 
					if Generate_random_diffs:
						imagePath = currentPath + outputFolder + "/" + outputName + "/" + outputName + "_training_data_" + str(training_data) + "/" + imageName
					else:
						imagePath = currentPath + outputFolder + "/" + outputName + "/" + outputName + "/" + imageName
					print(f"output file = \t\t{imagePath}")  
				if 'zone axis:' in line:
					zone = line[line.index(':')+1:-1].strip(' ')
					zone = list(zone.split(" "))
					zone = [float(i) for i in zone]
					print(f"zone axis= \t\t{zone}") 
				if 'gx vector:' in line:
					gx = line[line.index(':')+1:-1].strip(' ')
					gx = list(gx.split(" "))
					gx = [float(i) for i in gx]
					print(f"gx vector= \t\t{gx}")
				if 'surface normal:' in line:
					n = line[line.index(':')+1:-1].strip(' ')
					n = list(n.split(" "))
					n = [float(i) for i in n]
					print(f"surface normal = \t\t{n}")           
				if 'tilt offset (mrad):' in line:
					ktOffset = line[line.index(':')+1:-1].strip(' ')
					ktOffset = list(ktOffset.split(" "))
					ktOffset = [float(i) for i in ktOffset]
					print(f"tilt offset (mrad) = \t\t{ktOffset}") 
				#% Parameters for scattering path approximation
				if 'approximation:' in line:    # 0=Bloch wave, 1=kinematic, >1=Scatt. patch expansion
					approxOrder = line[line.index(':')+1:-1].strip(' ')
					approxOrder = int(approxOrder.split("%")[0].strip('\t').strip(' '))
					print(f"approximation = \t\t{approxOrder}")
				if 'term threshold:' in line:   # fraction of total intensity above which to include a term
					termThresh = line[line.index(':')+1:-1].strip(' ')
					termThresh = float(termThresh.split("%")[0].strip('\t').strip(' '))
					print(f"term threshold = \t\t{termThresh}") 
				if 'target Ug:' in line:   #  maximum value expected for any Ug (only used for term selection)
					targetUg = line[line.index(':')+1:-1].strip(' ')
					targetUg = float(targetUg.split("%")[0].strip('\t').strip(' '))
					print(f"target Ug = \t\t{targetUg}")
				if 'high tension:' in line:   #  maximum value expected for any Ug (only used for term selection)
					highTension = line[line.index(':')+1:-1].strip(' ')
					highTension = float(highTension.split("%")[0].strip('\t').strip(' '))
					print(f"high tension = \t\t{highTension}")
				if 'max g vector:' in line:   # gCutoff in 1/A.
					gCutoff = line[line.index(':')+1:-1].strip(' ')
					gCutoff = float(gCutoff.split("%")[0].strip('\t').strip(' '))
					print(f"max g vector = \t\t{gCutoff}")
				if 'max sg:' in line:   # sgCutoff, selects reflections that are closer than 0.02 1/A to the Ewald sphere.
					sgCutoff = line[line.index(':')+1:-1].strip(' ')
					sgCutoff = float(sgCutoff.split("%")[0].strip('\t').strip(' '))
					print(f"max sg = \t\t{sgCutoff}") 
				if 'thickness:' in line:   # thickness in A
					thickness = line[line.index(':')+1:-1].strip(' ')
					thickness =  float(thickness.split("%")[0].strip('\t').strip(' '))
					print(f"thickness = \t\t{thickness}")
				if 'scan compensation:' in line:   #scan Compensation as in the QED acquisition
					scanComp = line[line.index(':')+1:-1].strip(' ')
					scanComp =  float(scanComp.split("%")[0].strip('\t').strip(' '))
					print(f"scan compensation = \t\t{scanComp}")
				if 'tilt range (mrad):' in line:   
					tiltRange_mrad = line[line.index(':')+1:-1].strip(' ')
					tiltRange_mrad =  float(tiltRange_mrad.split("%")[0].strip('\t').strip(' '))
					print(f"tilt range (mrads) = \t\t{tiltRange_mrad}")
				if 'disc radius (pixels):' in line:   #disc diameter will be di = 2ri+1
					discRadius_pixel = line[line.index(':')+1:-1].strip(' ')
					discRadius_pixel =  int(discRadius_pixel.split("%")[0].strip('\t').strip(' '))
					print(f"disc radius (pixels) = \t\t{discRadius_pixel}")
				if 'detector size x:' in line: 
					detectorNx = line[line.index(':')+1:-1].strip(' ')
					detectorNx =  int(detectorNx.split("%")[0].strip('\t').strip(' '))
					detectorNy = detectorNx
					print(f"detector size x = \t\t{detectorNx}")
					
		return mode,cfgName,outputName,outputPath,imageName,imagePath,zone,gx,n,ktOffset,approxOrder,termThresh,highTension,gCutoff,sgCutoff,thickness,scanComp,tiltRange_mrad,discRadius_pixel,detectorNx,detectorNy

	mode = 'LARBED_SIM' #LARBED_SIM by default

	for training_data in range(N_training_data):

		if mode == 'LARBED_SIM':
		
			try:
				#read the parameter file which assigns the relevant variables
				mode,cfgName,outputName,outputPath,imageName,imagePath,zone,gx,n,ktOffset,approxOrder,termThresh, \
				highTension,gCutoff,sgCutoff,thickness,scanComp,tiltRange_mrad,discRadius_pixel,detectorNx,detectorNy \
				= read_parameter_file(filename,training_data)
			except:
				print('no file found - running with default parameters')

			
			if UseInputUgMasterList:
				InputUgRaw = InputUg / scaling #scaling factor for GP so it is not close to zero		
				InputUg = np.array(range( np.int((InputUgRaw.size - 1)/2) ), dtype=complex)			
				thickness = InputUgRaw[-1] * 1000 #this *1000 is scaling back up to the correct value,the thickness, as found by the NN or GP. It is important to have the variables in the same range.
				InputUg.real = InputUgRaw[0:-1:2]
				InputUg.imag = InputUgRaw[1:-1:2]

				#testing without updating thickness
				#InputUg = np.array(range( np.int((InputUgRaw.size)/2) ), dtype=complex)
				#print(f'InputUg={InputUg.shape} and \nInputUgRaw=\n{InputUgRaw}')
				#InputUg.real = InputUgRaw[::2] #for when thickness is not being varied and taken as last index of UgInput
				#InputUg.imag = InputUgRaw[1::2]
				print(f"thickness={thickness}")    
				print(f"InputUg = {InputUg.shape}")
								
			# read the crystal structure from a .cfg file:
			Xtal = pq.Crystal(cfgName)

			print(f"\n* QEDA: Simulation of {cfgName}, zone = {zone}\n")
			############################################/
			# Now we can allocate and initialize the pattern collection (e.g. LARBED data set).
			# This also assigns it the crystal object containing lattice parameters, etc.
			dataSet = pq.PatternCollection(Xtal,highTension,zone,n,gx,gCutoff,sgCutoff)

			#   # If we want to do simulations we can automatically create a disc pattern with a certain radius
			#   # and a given number of pixels spanning the radius:
			dataSet.GenerateDisc( tiltRange_mrad , discRadius_pixel , ktOffset)

			# For curiosity, we can look how many beams have been selected for each pattern
			dataSet.PrintTilts()
			# dataSet.CreateBeamCountMap()

			# Next, we can generate the structure factor master list
			nBeamsTotal = dataSet.CreateUgMasterList()
			print(f"\n\nAll patterns together contain {nBeamsTotal} different beams\n")

			# if we use the scattering path expansion, then convergence will be slowed down by the large value of U0.
			# We therefore simply set U0 to zero:
			if Generate_random_diffs:
				thickness = np.random.randint(100,high=1000)
				RandomUgMasterList.append( dataSet.GetRandomUgMasterList(0.2,0.4) )#Sherjeel implementation of random vectors(similar but with Ug limits
				RandomUgMasterList.append(thickness/1000)
				print(f'\n\nRandom Ug List {training_data} = \n\n{RandomUgMasterList}\n')
				dataSet.ComputeUgMasterList()
				dataSet.PrintUgMasterListVoid() 

			if UseInputUgMasterList:
				dataSet.UseInputUgMasterList(InputUg)
				print(f'InputUg:\n{InputUg}\n')
				dataSet.PrintUgMasterListVoid()
			else:
				dataSet.ComputeUgMasterList()      #for a normal simulation this would be used instead of generating a random set of Ug


		


			if approxOrder !=  0:
				dataSet.resetU0()
				if computeTermMaps == 1:
					for q in range(1,abs(approxOrder)):
						dataSet.ComputeDiffraction(thickness,-q,termThresh,targetUg)
						print(f"{mapName},termMap_q={q}.img")
						if Generate_random_diffs:
							random_diffs.append(dataSet.CreateDiffPat(mapName,detectorNx,detectorNy,scanComp) ) 							
						else:
							diff_pattern.append( dataSet.CreateDiffPat(mapName,detectorNx,detectorNy,scanComp) )          

			# # for debugging purposes only:
			dataSet.PrintUgMasterListVoid()
			if (approxOrder >= 0):
				# And now, we can start the dynamical calculation.
				dataSet.ComputeDiffraction(thickness,approxOrder,termThresh,targetUg)

				# # Let's save the data in binary format, so that we can analyze it later:
				dataSet.SaveDiffractionIntensities( imagePath )


				# # Finally, we ant to display the pattern.
				if Generate_random_diffs:
					random_diffs.append( dataSet.CreateDiffPat( detectorNx,detectorNy,scanComp) ) #this returns a variable of the diffraction pattern
				else:        
					diff_pattern.append( dataSet.CreateDiffPat( detectorNx,detectorNy,scanComp) ) #there is also a function to save to file



			# write all the simulated intensities, matrix diagonals, and Ar's to disk 
			dataSet.WriteSimulationParameters(outputPath)
			dataSet.PrintUgMasterList(outputPath  + "/_BeamList.txt" , 1)
			dataSet.PrintUgMasterList(outputPath  + "/_UgMasterList.txt" , 0)

			###############################################/
			# let's also fit kinematic rocking curves to get a first estimate of the structure factors:
			# First, we need to load the experimental data:
			dataSet.LoadDiffractionIntensities( imagePath  )
			nBeamsTotal = dataSet.CreateUgMasterList()
			#print(f"\nAll patterns together contain {nBeamsTotal} different beams\n")
			# Now, we can fit the diffraction data, first with kinematic theory, 
			# then with higher order approximations to dynamical theory: 
			# This function will populate the real parts of those Ugs in the masterUg 
			# array which are represented by actual diffraction intensities.
			errVal = dataSet.FitKinematicRockingCurves(thickness)
			print(f"Residual error after kinematic rocking curve fit: {errVal}\n")
			dataSet.PrintUgMasterList( (outputPath + "/_UgMasterList_KinematicFit.txt") , 0 )   
			

			
			
			
		if mode == 'LARBED_REC':
			print("* QEDA: Reconstruction from file %s\n",imagePath)

			############################################/
			#First, we need to load the experimental data:
			dataSet.LoadDiffractionIntensities(Image)
			# The crystal unit cell parameters have already been loaded (the actual atom positions will not be used)
			#Next, we can generate the structure factor master list (we won't compute them yet):
			dataSet.PrintTilts()

			nBeamsTotal = dataSet.CreateUgMasterList()
			print(f"All patterns together contain {nBeamsTotal} different beams\n")

			############################################/
			#Now, we can fit the diffraction data, first with kinematic theory, 
			#then with higher order approximations to dynamical theory: 
			# This function will populate the real parts of those Ugs in the masterUg 
			#array which are represented by actual diffraction intensities.
			errVal = dataSet.FitKinematicRockingCurves(thickness)
			print("Residual error after kinematic rocking curve fit: %f\n",errVal)

			#Just for debugging purposes, to see if starting with the correct solution we will also stay there:
			#dataSet.ComputeUgMasterList();
			dataSet.resetU0()
			dataSet.PrintUgMasterListVoid()

			if (approxOrder > 1):
				case = 0
				if case == 0:
					for i in range(50): 
						print("\n\n***** Iteration %d *****\n",i)
						errVal = dataSet.RefineStructureFactorsDynamicOnly(2,100,thickness)
						print("Residual error after double scattering rocking curve fit to dynamical part only: %f\n",errVal)
						errVal = dataSet.RefineStructureFactors(2,5,thickness)
						print("Residual error after double scattering rocking curve fit: %f\n",errVal)
						errVal = dataSet.RefineStructureFactorsLocal(2,15,thickness)
						print("Residual error after multivariate double scattering rocking curve fit: %f\n",errVal)
						errVal = dataSet.RefineStructureFactors(2,5,thickness)
						print("Residual error after double scattering rocking curve fit: %f\n",errVal)

				if case == 1:
					errVal = dataSet.RefineStructureFactors(2,100,thickness)
					print("Residual error after double scattering rocking curve fit: %f\n",errVal)
					errVal = dataSet.RefineStructureFactorsLocal(2,100,thickness)
					print("Residual error after multivariate double scattering rocking curve fit: %f\n",errVal)

				if case == 2:
					# dataSet.PrintUgMasterList();
					errVal = dataSet.RefineStructureFactors(2,100,thickness)
					print("Residual error after double scattering rocking curve fit: %f\n",errVal)

					dataSet.PrintUgMasterList()
					dataSet.SetDk(tiltRange_mrad,discRadius_pixel)
					dataSet.CreateDiffPat("temp2.img",detectorNx,detectorNy,scanComp)


				#Xtal.Print();
		
			
		if UseInputUgMasterList:
			UgReal = np.loadtxt( outputPath  + "/_UgMasterList.txt",skiprows=1,usecols=1 )			
			UgImag = np.loadtxt( outputPath  + "/_UgMasterList.txt",skiprows=1,usecols=2 )
			OutputUg = 0*(np.arange((UgReal.size)*2 +1  , dtype=float)) #arange excludes zero element so need to add +1 and we need another 
			OutputUg[-1] = thickness/1000 #this *1000 is scaling back up to the correct value,the thickness, as found by the NN or GP. It is important to have the variables in the same range.
			OutputUg[0:-1:2]= np.asarray(UgReal)
			OutputUg[1:-1:2] = np.asarray(UgImag)
		
		
		else:
			UgReal = np.loadtxt( outputPath  + "/_UgMasterList.txt",skiprows=1,usecols=1 )			
			UgImag = np.loadtxt( outputPath  + "/_UgMasterList.txt",skiprows=1,usecols=2 )
			OutputUg = 0*(np.arange((UgReal.size)*2 +1  , dtype=float)) #arange excludes zero element so need to add +1 and we need another 
			# print(f'\nUg of size:{UgReal.size} is \n {UgReal}')
			# print(f'\narange Ug :{np.arange((UgReal.size)*2 +1 )}')
			# print(f'OutputUg of size:{OutputUg.size} is \n {OutputUg}')
			OutputUg[-1] = thickness/1000 #this *1000 is scaling back up to the correct value,the thickness, as found by the NN or GP. It is important to have the variables in the same range.
			OutputUg[0:-1:2]= np.asarray(UgReal)
			OutputUg[1:-1:2] = np.asarray(UgImag)

	#this will display the random diffraction patterns
	def nlogf(x):
		return np.log(1+1000*x)
	nlog = np.vectorize(nlogf)
		
	if Generate_random_diffs:
		plt.imshow(nlogf(np.array(random_diffs[training_data]).reshape(detectorNx, detectorNy)))
		plt.title("Random Diffraction 1")
		plt.show()
		
		# try:
		# 	plt.imshow(nlogf(np.array(random_diffs[1]).reshape(detectorNx, detectorNy)))
		# 	plt.title("Random Diffraction 2")
		# 	plt.show()
		# except:
		# 	print("")

	else:
		plt.imshow(nlogf(np.array(diff_pattern[0]).reshape(detectorNx, detectorNy)))
		plt.title(outputName)
		plt.show()	

	if Generate_random_diffs:
		return RandomUgMasterList, random_diffs
	
	if ReturnOutputUg:
		return OutputUg[0:-1] * scaling, np.asarray(diff_pattern[0]/np.max(diff_pattern[0]))
	else:
		return np.asarray(diff_pattern[0]/np.max(diff_pattern[0]))
	



