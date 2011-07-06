/*
 * $Id: PVFile.h 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVCORE_FILE_H
#define PVCORE_FILE_H

#include <QString>
#include <QFile>
#include <QTextCodec>

#include <pvcore/general.h>

namespace PVRush {
	class LibExport File {
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

	LibExport QString file_get_cache_dir();
};

#endif	/* PVCORE_FILE_H */
