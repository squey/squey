# -*- coding: utf-8 -*-
#
# @file
#
# @copyright (C) Picviz Labs 2012-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

import time

def load_investigation(path):
	clickButton(waitForObject(":new-solution1.pvi[*] — Picviz Inspector 3.0.0.Open an investigation..._QPushButton_2"))
	sendEvent("QKeyEvent", waitForObject(":Name:_KUrlComboBox"), QEvent.KeyPress, 16777249, 0, 0, "", False, 1)
	time.sleep(1) # Needed to get the text typed correctly...
	type(waitForObject(":Name:_KUrlComboBox"), path)
	clickButton(waitForObject(":Load an investigation....Open_KPushButton"))

def import_source(path):
	clickButton(waitForObject(":new-solution1.pvi[*] — Picviz Inspector 3.0.0.Import files..._QPushButton_2"))
	sendEvent("QKeyEvent", waitForObject(":fileNameEdit_QLineEdit"), QEvent.KeyPress, 16777249, 0, 0, "", False, 1)
	time.sleep(1) # Needed to get the text typed correctly...
	type(waitForObject(":fileNameEdit_QLineEdit"), path)
	clickButton(waitForObject(":Import file.Open_QPushButton"))