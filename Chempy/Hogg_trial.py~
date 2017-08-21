#! /usr/bin/python
from .score_function import Hogg_bash
import sys
index = int(sys.argv[1]) # Beta index
print('Starting process %d' %(index))
Hogg_bash(index)

print('Stitching together')
if index == 1:
	from .score_function import Hogg_stitch
	Hogg_stitch()
print('Complete')