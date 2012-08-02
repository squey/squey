/**
 * \file PVFile.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_FILE_H
#define PVCORE_FILE_H

#include <QString>
#include <QFile>
#include <QTextCodec>

#include <pvkernel/core/general.h>

namespace PVRush {
	class LibKernelDecl File {
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

	LibKernelDecl QString file_get_cache_dir();
};

#endif	/* PVCORE_FILE_H */
