#!/usr/bin/env python
# Generate update files that will be downloaded by
# the inspector in order to check for new versions
#
# The "versions file" has the following format :
# 2.1.x
# 2.2.x
# ...
# where each line represents a major/minor version, and 'x' the latest patch version
#
# It will generate update-maj.min (one per branch) files with this format :
# maj
# min
# patch
# last-maj
# last-min
# last-patch
# where last-xxxx is the maj/min/patch version of the latest picviz version

import sys,os

def str_to_version(v):
	numbers = v.split('.')
	if (len(numbers) != 3):
		return None
	return [int(x) for x in numbers]

def version_n(v):
	return v[0]<<16|v[1]<<8|v[2]

if len(sys.argv) <= 2:
	print >>sys.stderr, "Usage: %s versions_file output_directory"
	sys.exit(1)

vfile = sys.argv[1]
outdir = sys.argv[2]

# Parse the versions file and find out the latest version
with open(vfile, "r") as f:
	versions_tmp = f.readlines()
versions = []
last_version = None
for v in versions_tmp:
	v = v[:-1].strip()
	if len(v) > 0:
		v = str_to_version(v)
		if (v == None):
			continue
		versions.append(v)
		if last_version == None or version_n(v) > version_n(last_version):
			last_version = v


# We have the different versions, 

for v in versions:
	major,minor,patch = v
	with open(os.path.join(outdir, "update-%d.%d" % (major,minor)), "w") as f:
			f.write(str(major)+"\n")
			f.write(str(minor)+"\n")
			f.write(str(patch)+"\n")
			f.write(str(last_version[0])+"\n")
			f.write(str(last_version[1])+"\n")
			f.write(str(last_version[2])+"\n")

with open(os.path.join(outdir, "update"), "w") as f:
	f.write("%d.%d.%d" % (last_version[0], last_version[1], last_version[2]))


