import os
#
# @file
#
# @copyright (C) Picviz Labs 2012-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

import pprint

pprint.pprint(os.environ)

INSPECTOR_FILES_DIR = os.path.join(os.environ.get("INSPECTOR_TESTS_DIR"), "files")