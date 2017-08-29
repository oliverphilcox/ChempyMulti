#! /usr/bin/python

## CODE TO RUN HOGG SCORES FOR VARIOUS ELEMENTS
## INCORRECT - THIS DOES NOT USE NEURAL NETWORKS!!! 
import fileinput
import os 
import sys
os.system('rm -rf BatchScoresNoScBay/')
os.system('mkdir BatchScoresNoScBay/')
os.system('rm -rf Scores/')
os.system('mkdir Scores')

print("1: Default Bay no Sc Score")

for line in fileinput.input("Chempy/parameter.py", inplace=True):
	if "\tyield_table_name_sn2_index" in line:
		print("\tyield_table_name_sn2_index = 5")
	elif "\tyield_table_name_agb_index" in line:
		print("\tyield_table_name_agb_index = 2")
	elif "\tyield_table_name_1a_index" in line:
		print("\tyield_table_name_1a_index = 2") 
	else:
		print(line,end='')
fileinput.close()		

from Chempy.score_function import Bayes_wrapper
Bayes_wrapper()
os.system('mkdir BatchScoresNoScBay/Default')
os.system('scp Scores/* BatchScoresNoScBay/Default/')
os.system('rm -rf Scores/')
os.system('mkdir Scores')

print("2: Chieffi Hogg no Sc Score")

for line in fileinput.input("Chempy/parameter.py", inplace=True):
	if "\tyield_table_name_sn2_index" in line:
		print("\tyield_table_name_sn2_index = 4")
	elif "\tyield_table_name_agb_index" in line:
		print("\tyield_table_name_agb_index = 2")
	elif "\tyield_table_name_1a_index" in line:
		print("\tyield_table_name_1a_index = 2") 
	else:
		print(line,end='')
		
fileinput.close()		
from Chempy.score_function import Bayes_wrapper
Bayes_wrapper()
os.system('mkdir BatchScoresNoScBay/Chieffi')
os.system('scp Scores/* BatchScoresNoScBay/Chieffi/')
os.system('rm -rf Scores/')
os.system('mkdir Scores')

print('All processes complete. Outputs are in BatchScoresNoScBay/ folder')