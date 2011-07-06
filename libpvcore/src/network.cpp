/*
 * $Id: network.cpp 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#include <QString>

#include <pvcore/network.h>

char *PVCore::network_ipntoa(const ip_addr_t addr)
{
	struct in_addr addr_source;

#ifdef WIN32
	addr_source.s_addr = (unsigned long)addr;
#else
	addr_source.s_addr = (in_addr_t)addr;
#endif
	return inet_ntoa(addr_source);
}

PVCore::Network::Network()
{

}

PVCore::Network::Network(char *addr)
{
  this->address = QString(addr);
}

PVCore::Network::Network(QString addr)
{
  this->address = addr;
}

PVCore::Network::~Network()
{

}

int PVCore::Network::is_ip_addr()
{
  QChar c = address.at(0);

  if ((c.isDigit()) && ( c < 51 )) {
    return 1;
  }

  return 0;
}

