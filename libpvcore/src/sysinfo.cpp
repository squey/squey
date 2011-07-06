/*
 * $Id: sysinfo.cpp 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#include <QFile>
#include <QDir>
#include <QList>
#include <QStringList>

#include <iostream>

#include <pvcore/debug.h>
#include <pvcore/sysinfo.h>

#include <stdlib.h>


PVCore::SysInfo::SysInfo()
{
}

PVCore::SysInfo::~SysInfo()
{
}

QStringList PVCore::SysInfo::get_card_from_id(QString id)
{
	QString pcifile_str;
	QByteArray data;
	QString sharedir;
	QDir dir;
	QStringList list;

	bool read_current_vendor = 0;

        sharedir = QString(getenv("PVCORE_SHARE_DIR"));
        if (sharedir.isEmpty()) {
	  sharedir = QString(PVCORE_SHARE_DIR);
        }

	pcifile_str.append(sharedir);
	pcifile_str.append(dir.separator());
	pcifile_str.append(QString("pci"));
	pcifile_str.append(dir.separator());
	pcifile_str.append(QString("pci.ids"));

	QFile pcifile(pcifile_str);

	pcifile.open(QIODevice::ReadOnly | QIODevice::Text);


	data = pcifile.readLine();
	while (!data.isEmpty()) {
		data.chop(1);

		QString data_str(data);

		if (!data_str.startsWith("#")) {
			if (!data_str.startsWith("\t")) {
				QStringList data_list = data_str.split(" ");
	
				if (id.startsWith(data_list[0])) {
				  	// Since our ID starts with the same chars than the vendor string, then we have it!
					if (!data_str.isEmpty()) {
						data_str.remove(0,6);
				  		list << data_str;
				  	}
					read_current_vendor = 1;
				} else {
					read_current_vendor = 0;
				}
			} else {
				if (!data_str.startsWith("\t\t")) {
					QStringList data_list = data_str.split(" ");
					data_list[0].remove(0,1);

					if (id.endsWith(data_list[0]) && (read_current_vendor)) {
					  	// Since our ID end with the same chars than the device string, then we have it!
						data_str.remove(0,7);
					  	list << data_str;
					}
				}

			}
		}

		data = pcifile.readLine();
	}

	pcifile.close();

  return list;		

}

int PVCore::SysInfo::gpu_count()
{

	int gpu_counter = 0;
#ifdef LINUX
	QFile pcifile(LINUX_PCI_DEVICES_FILE);
	QByteArray data;

	pcifile.open(QIODevice::ReadOnly | QIODevice::Text);
	data = pcifile.readLine();

	while (!data.isEmpty()) {
		data.chop(1);

		QString data_str(data);
		QStringList strlist = data_str.split('\t');

		QStringList list = get_card_from_id(strlist[1]);
		// list[0] = vendor
		// list[1] = device

		// debug_qstringlist(list);

		data = pcifile.readLine();
	}

	pcifile.close();
	
	return 0;
#elif WIN32
	return 1;
#endif
}

//QList <QStringList> pvcore_device_graphical_cards_scan
