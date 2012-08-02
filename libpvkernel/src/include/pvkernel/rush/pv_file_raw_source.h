/**
 * \file pv_file_raw_source.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PV_FILE_RAW_SOURCE_H
#define PVCORE_PV_FILE_RAW_SOURCE_H

#include <QString>
#include <QStringList>


#include <pvkernel/core/general.h>



class LibKernelDecl PVFileRawSource {
	private:
		QString filename;

	public:
		PVFileRawSource(const QString &file_name);
		~PVFileRawSource();

		QString get_filename()const{return filename;}
		
		QStringList get_list();

};


#endif	/* PVCORE_PV_FILE_RAW_SOURCE_H */
