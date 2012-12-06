#!/usr/bin/env python

# \file diff_stdout.py
#
# Copyright (C) Picviz Labs 2010-2012

# Check the difference of the output between a saved one and the execution of a program (with its args)
# Usage: ./diff_stdout.py file_ref file_output_diff prog [args... ]

import sys,os,subprocess,tempfile

def get_script_path(): return os.path.dirname(os.path.realpath(os.path.abspath(__file__)))

file_ref = sys.argv[1]
file_out_diff = sys.argv[2]
file_out = file_out_diff + ".run"

prog = sys.argv[3:]

tmp_dir = tempfile.mkdtemp()
def mksfifo(name):
	f = os.path.join(tmp_dir, name)
	try:
		os.mkfifo(f)
	except OSError as e:
		print >>sys.stderr, "Fail to create FIFO: %s" % e
		sys.exit(4)
	return f


# Create diff output's directory if necessary
file_out_diff_dir = os.path.realpath(os.path.dirname(file_out_diff))
try:
	os.makedirs(file_out_diff_dir)
except OSError as e:
	if (e.errno != 17): # Path already exists
		raise

with open(file_out, "w") as f:
	p = subprocess.Popen(prog, stdout=f)
	ret = p.wait()
	if (ret != 0):
		#os.remove(file_out)
		print >>sys.stderr, "Error running program '%s', return code is %d." % (" ".join(prog), ret)
		sys.exit(ret)

ret = 0
with open(file_out_diff, "w") as f:
	p = subprocess.Popen(["diff", "-u", file_ref, file_out], stdout=f, env={"LANG": "C"})
	ret = p.wait()

#os.remove(file_out)
if (ret != 0):
	with open(file_out_diff, "r+") as f:
		line_str = f.readline()

	if (line_str == "Binary files %s and %s differ\n" % (file_ref, file_out)):
		with open(file_out_diff, "w+") as f:
			p = subprocess.Popen([os.path.join(get_script_path(), "diff_hexdump.sh"), file_ref, file_out], stdout=f)
			p.wait()

	print >>sys.stderr, "Output differs for program '%s'. stdout (dumped in '%s') has been compared against '%s', and the differences are in '%s'." % (" ".join(prog), file_out, file_ref, file_out_diff)
else:
	os.unlink(file_out_diff)
	os.unlink(file_out)

try:
	shutil.rmtree(tmp_dir)
except: pass

sys.exit(ret)
