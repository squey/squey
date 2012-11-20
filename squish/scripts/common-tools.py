# -*- coding: utf-8 -*-
import re
import os

def strcmp_rx(str, rx):
    ''' Returns true if 'str' matches 'rx' '''
    return re.match(rx, str) != None

def default_project_name(number):
    return PROJECT_DEFAULT_NAME % number

def mainwindow_name_regexp_for_project(project_name):
    return "^%s\[\*\] — Picviz Inspector \d+\.\d+\.\d+" % project_name

def path_files(file):
    return os.path.join(SQUISH_FILES, file)