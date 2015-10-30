/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_FILE_H
#define PVCORE_FILE_H

#include <QString>
#include <QFile>
#include <QTextCodec>

#include <pvkernel/core/general.h>

namespace PVRush {
	class File {
	public:
		File(QString filename);
		~File();
		
		int Uncompress(QString srcfile, QString dstfile);


		QString cached_filenameroot;
		QFile file;
		QString name;
		QTextCodec *codec;
		int is_compressed;
	};
};

#endif	/* PVCORE_FILE_H */
