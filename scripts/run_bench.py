#!/usr/bin/python

""" Script to run benchmarks and send results on the codespeed server.  """

from collections import namedtuple
import json
import numpy
import os
import socket
import subprocess
import urllib
import urllib2


Measure = namedtuple("Measure", ["mean", "max", "min", "median", "std"])
CODESPEED_URL = 'http://codespeed.srv.picviz/'


def get_git_hash():
    """ Get hash of the current git HEAD. """
    return subprocess.check_output("git rev-parse HEAD", shell=True)


def get_git_branch():
    """ Get the current git branch. """
    return subprocess.check_output("git rev-parse --abbrev-ref HEAD",
                                   shell=True).strip()


def run(program_name, num_run):
    """ Run the program and measures its performances. """
    times = numpy.array(
        [float(subprocess.check_output(program_name).split()[-1])
         for _ in xrange(num_run)])

    return Measure(mean=numpy.mean(times), max=numpy.max(times),
                   min=numpy.min(times), median=numpy.median(times),
                   std=numpy.std(times))


def add_to_codespeed(data):
    """ Add data to the codespeed server. """
    response = "None"
    try:
        f = urllib2.urlopen(
            CODESPEED_URL + 'result/add/json/', urllib.urlencode(data))
    except urllib2.HTTPError as e:
        print str(e)
        print e.read()
        return
    response = f.read()
    f.close()
    print "Server (%s) response: %s\n" % (CODESPEED_URL, response)


def main(program_name, num_run):
    """ Run the program and save data in codespeed. """
    if get_git_branch() != "master":
        raise ValueError("Bench can be run only from master branch")
    measure = run(program_name, num_run)

    # Create JSon information
    formated_measure = {
        "commitid": get_git_hash(),
        "project": "inspector",
        "branch": "master",
        "executable": "release",  # We may check for this cmake mode
        "benchmark": os.path.basename(program_name),
        "environment": socket.gethostname().split('.')[0],
        "result_value": measure.mean,
        "std_dev": measure.std,
        "mean": measure.median,
        "min": measure.min,
        "max": measure.max,
        "median": measure.median
    }
    data = {'json': json.dumps([formated_measure])}
    # Push them on the server
    add_to_codespeed(data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run benchmarks and send '
                                     'result on the codespeed server')
    parser.add_argument('program_path', type=str,
                        help='Absolute path to the program to run')
    parser.add_argument('--num_run', default=10, type=int,
                        help='Number of time we run tests to make sure result '
                        'is reliable')

    args = parser.parse_args()

    main(args.program_path, args.num_run)
