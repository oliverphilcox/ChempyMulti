#! /usr/bin/python
from Chempy.score_function import Hogg_bash
import sys
index = int(sys.argv[1]) # Beta index
print('Starting process %d' %(index))
Hogg_bash(index)

if index == 19:
	print('Stitching together')
	from Chempy.score_function import Hogg_stitch
	Hogg_stitch()
	print('Complete')
