/*
 * $Id: sysinfo.h 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVCORE_SYSINFO_H
#define PVCORE_SYSINFO_H

#include <QStringList>
#include <QString>

#include <pvcore/general.h>

#define LINUX_PCI_DEVICES_FILE "/proc/bus/pci/devices"

namespace PVCore {
	class LibCoreDecl SysInfo {
	public:
		SysInfo();
		~SysInfo();

		QStringList get_card_from_id(QString id);
		int gpu_count();
	};
}

#endif	/* PVCORE_SYSINFO_H */

