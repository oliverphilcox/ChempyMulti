#! /usr/bin/python
from Chempy.score_function import Hogg_bash
import sys
index = int(sys.argv[1]) # Beta index
Hogg_bash(index)

if index == 1:
	from Chempy.score_function import Hogg_stitch
	Hogg_stitch