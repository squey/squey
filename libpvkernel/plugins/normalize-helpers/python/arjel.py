#!/usr/bin/python3.2
# ARJEL XML file parser
#
import xml.etree.ElementTree,sys,pprint
import csv

## Global vars

# XML tree
xml_tree = None
all_fields = ['action_tag', 'Ad', 'BonusApres', 'BonusApresMise', 'BonusAvant', 'BonusAvantMise', 'BonusMouvement', 'BonusNom', 'CP', 'CauseRefus', 'Civilite', 'Clair', 'Compte:Max', 'Compte:Min', 'Compte_number', 'DateDemande', 'DateEffective', 'DateEvt', 'DateHeure', 'DateModif', 'DateN', 'DepotMax:MontantMM', 'DepotMax:PeriodeMM', 'DepotMax:TypeJeuMM', 'DepotMax_number', 'DescPari:Combi', 'DescPari_number', 'DptN', 'Duree', 'Email', 'Gain', 'IDCoffre', 'IDEvt', 'IDJoueur', 'IDOper', 'IDSession', 'IPJoueur', 'Info', 'Lig:Clair', 'Lig:DateHeure', 'Lig:No', 'Lig:Tech', 'Lig:Type', 'Lig_number', 'Login', 'Mise', 'MiseMax:MontantMM', 'MiseMax:PeriodeMM', 'MiseMax:TypeJeuMM', 'MiseMax_number', 'MontantRembours', 'MoyenPaiement', 'Nom', 'Pari:Choix', 'Pari:Cote', 'Pari_number', 'Pays', 'PaysN', 'Prenoms', 'Pseudo', 'SoldeApres', 'SoldeApresGain', 'SoldeApresMise', 'SoldeApresRembours', 'SoldeAvant', 'SoldeAvantGain', 'SoldeAvantMise', 'SoldeAvantRembours', 'SoldeClos', 'SoldeMouvement', 'Tech', 'TelFixe', 'TelMob', 'TypAg', 'TypeMoyenPaiement', 'TypeRefus', 'Ville', 'VilleN']

# Current lot_id
lot_id = 0

first = True

ACTION_TAG = "action_tag"

def print_format_from_fields(f, fields):
	f.write("<?xml version='1.0' encoding='UTF-8'?>\n")
	f.write("<!DOCTYPE PVParamXml>\n")
	f.write('<param version="3">\n')
	for field in fields:
		f.write('<field>\n')
		f.write("<axis mapping=\"default\" type=\"enum\" name=\"%s\" plotting=\"default\" />\n" % field)
		f.write('</field>\n')
	f.write('</param>\n')

def picviz_open_file(path):
	global xml_tree,all_fields
	xml_tree = xml.etree.ElementTree.parse(path)
	#all_fields = list(get_fields_from_tree(xml_tree))
	#all_fields.sort()
	#all_fields.insert(0, ACTION_TAG)
	
	#with open("arjel-picviz.format", "w") as f:
	#	print_format_from_fields(f, all_fields)


def picviz_seek_begin():
	pass

def picviz_pre_discovery():
	return True;

def picviz_is_element_text():
	return True;

def picviz_get_next_chunk(min_chunk_size):
	global xml_tree,first,all_fields
	if not first:
		return list()
	else:
		first = False

	ret_elts = list()
	elements = get_elements_from_tree(xml_tree)
	for e in elements:
		ret_e = list()
		if len(e) > len(all_fields):
			print("(Python ARJET) WARNING: element with %d fields (> %d)", len(e), len(all_fields))
		for f in all_fields:
			if f in e: ret_e.append(e[f])
			else: ret_e.append("")
		ret_elts.append(ret_e)

	return ret_elts

def picviz_close():
	pass

def iterparent(tree):
	for parent in tree.iter():
		for child in parent:
			yield parent, child

def get_fields_from_tree(xml_tree):
	all_fields = set()
	for lot in xml_tree.getiterator("Lot"):
		for action in lot:
			for parent,elt in iterparent(action):
				if elt == action: continue
				if len(elt) > 0:
					all_fields.add(elt.tag + "_number")
				else:
					ctag = elt.tag
					if (parent != action):
						ctag = parent.tag + ":" + ctag
					all_fields.add(ctag)
	return all_fields

def get_elements_from_node(node, add_parent_tag):
	elements = list()
	node_fields = dict()
	# Get all fields of this node
	for child in node:
		if len(child) == 0:
			ctag = child.tag
			if (add_parent_tag):
				ctag = node.tag + ":" + ctag
			if child.text == None:
				child.text = ""
			node_fields[ctag] = str(child.text)
	
	# Process other children
	children_tags = set()
	for child in node:
		if len(child) > 0:
			children_tags.add(child.tag)
	for ctag in children_tags:
		ctag_id = 0
		for cchild in node.findall(ctag):
			elements_c = get_elements_from_node(cchild, True)
			# Update children's elements with our fields
			ctag_field = ctag + "_number"
			for e in elements_c:
				e.update(node_fields)
				e.update({ctag_field: str(ctag_id)})
			elements.extend(elements_c)
			ctag_id = ctag_id + 1
	
	if len(elements) == 0:
		elements = [node_fields]
	
	return elements

def get_elements_from_tree(xml_tree):
	elements = list()
	for lot in xml_tree.getiterator("Lot"):
		for action in lot:
			action_elts = get_elements_from_node(action, False)
			for e in action_elts:
				e[ACTION_TAG] = action.tag
			elements.extend(action_elts)

	return elements

picviz_open_file(sys.argv[1])
elements = picviz_get_next_chunk(0)

csv_writer = csv.writer(sys.stdout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#csv_writer.writerow(all_fields)
for e in elements:
	csv_writer.writerow(e)

#pprint.pprint(get_fields("/media/truecrypt1/arjel-1"))
#pprint.pprint(get_elements_from_path("/media/truecrypt1/arjel-1-extract"))

#picviz_open_file("/media/truecrypt1/arjel-1")
#print(all_fields)
#pprint.pprint(picviz_get_next_chunk(0))
#pprint.pprint(picviz_get_next_chunk(0))
