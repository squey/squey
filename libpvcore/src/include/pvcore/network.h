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
        LibExport char *network_ipntoa(const ip_addr_t addr);

	class LibExport Network {
	private:
		QString address;

	public:
		Network();
		Network(char *addr);
		Network(QString addr);
		~Network();

		int is_ip_addr();
	};
}

#endif	/* PVCORE_NETWORK_H */
