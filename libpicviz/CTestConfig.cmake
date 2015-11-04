#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

SET(CTEST_PROJECT_NAME "libpicviz")
SET(CTEST_NIGHTLY_START_TIME "00:00:00 EST")
SET(CTEST_DROP_METHOD "http")

IF(CTEST_DROP_METHOD STREQUAL "http")
  SET(CTEST_DROP_SITE "cactus.wallinfire.net")
  SET(CTEST_DROP_LOCATION "/CDash/submit.php?project=libpicviz")
  SET(CTEST_TRIGGER_SITE "")
ENDIF(CTEST_DROP_METHOD STREQUAL "http")
