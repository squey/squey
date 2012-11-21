#!/bin/bash
export SQUISH_DIR=`pwd`
export SQUISH_SCRIPT_DIR="$SQUISH_DIR/scripts"
squishserver --config addAUT picviz-inspector.sh "`pwd`/.." 
cd "$SQUISH_DIR/test_suites"
squishserver --daemon
squishrunner --testsuite suite_test_suite_1 --testcase tst_correlations_menu
