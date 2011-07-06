/*
 * $Id: pv_file_raw_source.h 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 *
 */

#ifndef PVCORE_PV_FILE_RAW_SOURCE_H
#define PVCORE_PV_FILE_RAW_SOURCE_H

#include <QString>
#include <QStringList>


#include <pvcore/general.h>



class LibExport PVFileRawSource {
	private:
		QString filename;

	public:
		PVFileRawSource(const QString &file_name);
		~PVFileRawSource();

		QString get_filename()const{return filename;}
		
		QStringList get_list();

};


#endif	/* PVCORE_PV_FILE_RAW_SOURCE_H */
