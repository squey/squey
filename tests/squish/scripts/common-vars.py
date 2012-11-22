import os
import pprint

pprint.pprint(os.environ)

INSPECTOR_FILES_DIR = os.path.join(os.environ.get("INSPECTOR_TESTS_DIR"), "files")