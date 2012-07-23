/**
 * \file sysinfo.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_SYSINFO_H
#define PVCORE_SYSINFO_H

#include <QStringList>
#include <QString>

#include <pvkernel/core/general.h>

#define LINUX_PCI_DEVICES_FILE "/proc/bus/pci/devices"

namespace PVCore {
	class LibKernelDecl SysInfo {
	public:
		SysInfo();
		~SysInfo();

		QStringList get_card_from_id(QString id);
		int gpu_count();
	};
}

#endif	/* PVCORE_SYSINFO_H */

