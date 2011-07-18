/*
 * $Id: network.h 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVCORE_NETWORK_H
#define PVCORE_NETWORK_H

#include <QString>

#include <dnet.h>

#include <pvcore/general.h>

#ifdef WIN32
#include <Winsock2.h>
#else
#include <arpa/inet.h>
#endif

namespace PVCore {
	struct LibExport Network {
		static bool ipv4_aton(QString const& ip, uint32_t& ip_n);
		static char* ipv4_ntoa(const ip_addr_t addr);
	};
}

#endif	/* PVCORE_NETWORK_H */
