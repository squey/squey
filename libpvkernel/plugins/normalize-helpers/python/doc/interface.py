#!/usr/bin/python3.2
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

def inendi_open_file(path):
	pass

def inendi_seek_begin():
	pass

def inendi_pre_discovery():
	return True;

def inendi_is_element_text():
	return True;

def inendi_get_next_chunk(min_chunk_size):
	return [ ["a", "1"], ["b", "2"], ["c", 0], [False, True] ];

def inendi_close():
	pass
