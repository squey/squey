/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PV_FILE_RAW_SOURCE_H
#define PVCORE_PV_FILE_RAW_SOURCE_H

#include <QString>
#include <QStringList>


#include <pvkernel/core/general.h>



class PVFileRawSource {
	private:
		QString filename;

	public:
		PVFileRawSource(const QString &file_name);
		~PVFileRawSource();

		QString get_filename()const{return filename;}
		
		QStringList get_list();

};


#endif	/* PVCORE_PV_FILE_RAW_SOURCE_H */
