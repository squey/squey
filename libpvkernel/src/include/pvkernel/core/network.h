/**
 * \file network.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_NETWORK_H
#define PVCORE_NETWORK_H

#include <QString>

#include <pvkernel/core/general.h>
#include <pvkernel/core/dumbnet.h>

namespace PVCore {

struct LibKernelDecl Network {
	static bool ipv4_aton(QString const& ip, uint32_t& ip_n);
	static bool ipv4_aton(const char* buffer, size_t n, uint32_t& ret);
	static bool ipv4_a16ton(const uint16_t* buffer, size_t n, uint32_t& ret);
	static char* ipv4_ntoa(const ip_addr_t addr);
};

}

#endif	/* PVCORE_NETWORK_H */
