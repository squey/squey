export $SQUISH_DIR=`pwd`
export SQUISH_SCRIPT_DIR=$SQUISH_DIR/scripts
cd $SQUISH_DIR/test_suites
squishrunner --testsuite suite_test_suite_1 --testcase tst_correlations_menu
