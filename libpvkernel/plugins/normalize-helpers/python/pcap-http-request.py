#!/usr/bin/python3.2

# \file pcap-http-request.py
#
# Copyright (C) Picviz Labs 2010-2012

import dpkt

## Global vars
# PCAP file
pcap = None
first = True

def picviz_open_file(path):
	global pcap
#	print ("READING FROM: " + path)
	pcap_file = open(path, mode="rb")
#	print (str(pcap_file))
	pcap = dpkt.pcap.Reader(pcap_file)

def picviz_seek_begin():
	pass

def picviz_pre_discovery():
	return True;

def picviz_is_element_text():
	return True;

def picviz_get_next_chunk(min_chunk_size):
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

def picviz_close():
	pass


# if __name__ == "__main__":
# 	my_pcap = "/donnees/SVN/cactuslogs/pcap/waledac.pcap"
# 	picviz_open_file(my_pcap)
# 	picviz_get_next_chunk(0)

