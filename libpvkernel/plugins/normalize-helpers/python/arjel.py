#!/usr/bin/python3.2
# ARJEL XML file parser
#
# fields:
# action,idoper,dateevt,idevt,idjoueur,idsession,ipjoueur,idcoffre,tech,clair,dateheure,soldeavantrembours,montantrembours,soldeapresrembours,info
import xml.etree.ElementTree

## Global vars

# XML tree
xml_tree = None

# Current lot_id
lot_id = 0

first = True

def picviz_open_file(path):
	global xml_tree,xml_root
	xml_tree = xml.etree.ElementTree.parse(path)

def picviz_seek_begin():
	pass

def picviz_pre_discovery():
	return True;

def picviz_is_element_text():
	return True;

def picviz_get_next_chunk(min_chunk_size):
	global xml_tree,xml_root,first,lot_id
	if not first:
		return list()
	else:
		first = False
	elements = list()
	for lots in xml_tree.getiterator("Lot"):
		for action in lots:
			fields = list()
			fields.append(action.tag)
			fields.append(action.findtext("IDOper"))
			fields.append(action.findtext("DateEvt"))
			fields.append(action.findtext("IDEvt"))
			fields.append(action.findtext("IDJoueur"))
			
			elements.append(fields)

		lot_id = lot_id + 1

	return elements

def picviz_close():
	pass

#picviz_open_file("/data/arjel-1-extract")
#print(picviz_get_next_chunk(0))
