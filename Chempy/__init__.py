import os
import inspect

# for compatibility
import platform
if platform.system()=='Windows':
	string='\\'
else:
	string='/'

localpath = string.join(os.path.abspath(inspect.getfile(inspect.currentframe())).split(string)[:-1])
localpath += string
