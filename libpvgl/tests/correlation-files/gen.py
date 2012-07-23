#!/usr/bin/python

# \file gen.py
#
# Copyright (C) Picviz Labs 2010-2012

#

import sys,csv,os,random,struct,socket,string,math

if len(sys.argv) < 2:
	print >>sys.stderr, "Usage: %s n" % sys.argv[0]
	sys.exit(1)

random.seed()
n = int(sys.argv[1])

list_ips = list()
list_domains = list()
nelts = int(math.sqrt(n))
for i in xrange(0,nelts):
	iprand = random.randint(0, (1<<32)-2)
	ip_txt = socket.inet_ntoa(struct.pack('!I', iprand))
	domain = ''.join(random.choice(string.ascii_lowercase) for x in range(10)) + "." + random.choice(('org', 'com', 'fr', 'net', 'cz', 'ru'))
	list_ips.append(ip_txt)
	list_domains.append(domain)

csv_dns = csv.writer(open("dns.csv", "w+"), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
csv_proxy = csv.writer(open("proxy.csv", "w+"), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
csv_fw = csv.writer(open("fw.csv", "w+"), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
# Time IP Domain
for i in xrange(0,n):
	time = i*1000
	ip = list_ips[random.randint(0, nelts-1)]
	domain = list_domains[random.randint(0, nelts-1)]
	csv_dns.writerow((time, ip, domain))
	csv_proxy.writerow((time, ip, "http://www.%s.com/index.html" % domain))
	if (i%10000 == 0):
		csv_fw.writerow((time, ip, 8000))
	else:
		csv_fw.writerow((time, ip, random.choice(("80","443"))))

