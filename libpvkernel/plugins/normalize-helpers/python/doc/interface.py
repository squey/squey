#!/usr/bin/python3.2

# \file interface.py
#
# Copyright (C) Picviz Labs 2010-2012

def picviz_open_file(path):
	pass

def picviz_seek_begin():
	pass

def picviz_pre_discovery():
	return True;

def picviz_is_element_text():
	return True;

def picviz_get_next_chunk(min_chunk_size):
	return [ ["a", "1"], ["b", "2"], ["c", 0], [False, True] ];

def picviz_close():
	pass
