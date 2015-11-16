#!/usr/bin/python3.2
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

import dpkt

## Global vars
# PCAP file
pcap = None
first = True

def inendi_open_file(path):
	global pcap

	pcap_file = open(path, mode="rb")
	try:
		pcap = dpkt.pcap.Reader(pcap_file)
	except:
		pass

def inendi_seek_begin():
	pass

def inendi_pre_discovery():
	return True;

def inendi_is_element_text():
	return True;

def inendi_get_next_chunk(min_chunk_size):
	global pcap, first
	if not first:
		return list()
	else:
		first = False

	elements = list()
	for ts, buf in pcap:
		eth = dpkt.ethernet.Ethernet(buf)
		ip = eth.data
		tcp = ip.data

		if tcp:
			try:
				if tcp.dport == 80 and len(tcp.data) > 0:
					http = dpkt.http.Request(tcp.data)

					fields = list()
					fields.append(str(tcp.sport))
					fields.append(str(tcp.dport))
					fields.append(http.uri)

					elements.append(fields)

#					print(str(elements))
			except:
				pass

	print (str(elements))
	return elements

def inendi_close():
	pass


# if __name__ == "__main__":
# 	my_pcap = "/donnees/SVN/cactuslogs/pcap/waledac.pcap"
# 	inendi_open_file(my_pcap)
# 	inendi_get_next_chunk(0)

