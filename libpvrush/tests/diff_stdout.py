#!/usr/bin/env python
# Check the difference of the output between a saved one and the execution of a program (with its args)
# Usage: ./diff_stdout.py file_ref file_output_diff prog [args... ]

import sys,os,subprocess

file_ref = sys.argv[1]
file_out_diff = sys.argv[2]
file_out = file_ref + ".run"

prog = sys.argv[3:]


with open(file_out, "w") as f:
	p = subprocess.Popen(prog, stdout=f)
	ret = p.wait()
	if (ret != 0): sys.exit(ret)

with open(file_out_diff, "w") as f:
	p = subprocess.Popen(["diff", "-u", file_ref, file_out], stdout=f)
	ret = p.wait()
	os.unlink(file_out)
	if (ret != 0): sys.exit(ret)

os.unlink(file_out_diff)
sys.exit(0)
