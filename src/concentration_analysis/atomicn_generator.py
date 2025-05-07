'''code to count the number of atoms per component'''
# atom number(s) per component are counted and returned in
# a saved file

# import depdencies
import numpy as np
import sys
import scipy. constants as si
import openbabel.pybel as pybel # converting SMILES to pybel objects
import os

# user-defined variables start --------------------------------

# set path to results
res_path = str('/Users/user/Documents/GitHub/PyCHAM/' +
	'PyCHAM/output/ex_chem_scheme/Flow_Reactor_Example_Run_Output')

# path to PyCHAM
PyCHAM_path = str('/Users/user/Documents/GitHub/PyCHAM/PyCHAM')

# path to save to
save_path = str('/Users/user/Library/CloudStorage/OneDrive' +
	'-TheUniversityofManchester/SOAPRA/Lukas_Share')

file_name = str('CHON_test.txt')

# string of atoms to consider
atoms = 'CHON' 


# user-defined variables end --------------------------------

# define function
def an_generator(res_path, PyCHAM_path, save_path, atoms, file_name):

	# ensure PyCHAM can be seen
	sys.path.append(PyCHAM_path)
	import retr_out

	# create the self object so that results path is stored
	self = self_def(res_path)

	# import results
	for prog in retr_out.retr_out(self):
		prog = prog

	# get the SMILES strings of every component
	SMILES = (self.ro_obj.rSMILES)

	# prepare to hold CHON of each component
	atom_num = np.zeros((len(SMILES), len(atoms))).astype('int')

	# loop through SMILES to get atom number
	for compi in range(0, len(SMILES)):
		
		# if hydrogens are in SMILES string then remove
		if 'H' in SMILES[compi] or 'h' in SMILES[compi]:
			SMILES[compi] = SMILES[compi].replace('H4', '')
			SMILES[compi] = SMILES[compi].replace('H', '')
			SMILES[compi] = SMILES[compi].replace('h', '')

		# convert SMILES to pybel object
		Pybel_object = pybel.readstring('smi', SMILES[compi])

		for ai in range(len(atoms)):
			if atoms[ai] not in Pybel_object.formula:
				continue
			else:
				atom_num[compi, ai] = formula_check(Pybel_object.formula,
					atoms[ai])
	
	# make directory if not already existing
	if (os.path.isdir(str(save_path)) == False):
		os.mkdir(str(save_path))

	np.savetxt(str(save_path + '/' + file_name), atom_num, fmt='%i', delimiter='	')

		
	return()

# function to setup self
def self_def(dir_path_value):

	class testobj(object):
		pass

	self = testobj()
	self.dir_path = dir_path_value

	return(self)

# function to get atom number
def formula_check(form, atom):
	# use the pybel object formula attribute to get 
	# numbers of each atom
	aindx = form.index(atom)+1 # where atom number starts
	# where atom number ends
	aindxend = aindx+1
	
	if (aindx == len(form)): # if no number
		return(1)

	if ((form[aindx]).isnumeric() == False): # if no number
		return(1)

	while aindxend<len(form):
		if (form[aindxend]).isnumeric():
			aindxend += 1
		else:
			break
	
	
	return(int(float(form[aindx:aindxend])))

# call function
an_generator(res_path, PyCHAM_path, save_path, atoms, file_name)