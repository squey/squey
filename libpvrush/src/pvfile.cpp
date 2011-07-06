/*
 * $Id: pvfile.cpp 2976 2011-05-26 04:06:27Z dindinx $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#include <QByteArray>
#include <QResource>		// just because there is the function isCompressed() in it
#include <QLocale>

#include <pvrush/PVFile.h>

#include <archive.h>

#include <stdlib.h>

#ifdef WIN32
  #include <windows.h>
  #include <io.h>
#endif

using namespace PVCore;

/******************************************************************************
 *
 * PVRush::file_get_cache_dir
 *
 *****************************************************************************/
QString PVRush::file_get_cache_dir()
{
	QString cachedir;

	cachedir = QString(getenv("PVRUSH_CACHE_DIR"));
	if (cachedir.isEmpty()) {
		cachedir = QString(PVRUSH_CACHE_DIR);
	}

	return cachedir;
}

/******************************************************************************
 *
 * PVRush::File::File
 *
 *****************************************************************************/
PVRush::File::File(QString filename)
{
	QFile     file(filename);
	QResource resource(filename, QLocale());

	file.open(QIODevice::ReadOnly);

	QByteArray filedata = file.read(32);
	const QByteArray ba(filedata);

	name = filename;

	if (file.error() || (file.handle() < 0)) {
		PVLOG_ERROR("Cannot open file '%s', error: %d\n", filename.toUtf8().data(), file.error());
		return;
	}

	PVLOG_DEBUG("Creating file object from '%s', handler %d\n", filename.toUtf8().data(), file.handle());

	this->codec = QTextCodec::codecForUtfText(ba);
	this->is_compressed = resource.isCompressed();

}

/******************************************************************************
 *
 * PVRush::File::~File
 *
 *****************************************************************************/
PVRush::File::~File()
{
	this->file.close();
}

/******************************************************************************
 *
 * PVRush::File::Uncompress
 *
 *****************************************************************************/
int PVRush::File::Uncompress(QString srcfile, QString dstfile)
{
	int r;
	size_t size;
	QByteArray filename = srcfile.toUtf8();

	struct archive_entry *ae;
	char buff[8192];
	size_t buffsize = 8192;

	QFile output(dstfile);

	output.open(QIODevice::WriteOnly);

	struct archive *a = archive_read_new();
	archive_read_support_compression_all(a);
	archive_read_support_format_raw(a);
	r = archive_read_open_filename(a, filename.data(), 16384);
	if (r != ARCHIVE_OK) {
		/* ERROR */
	}
	r = archive_read_next_header(a, &ae);
	if (r != ARCHIVE_OK) {
		/* ERROR */
	}

	for (;;) {
		size = archive_read_data(a, buff, buffsize);
		if (size < 0) {
			/* ERROR */
		}
		if (size == 0)
			break;
		write(output.handle(), buff, size);
	}

	archive_read_finish(a);

	output.close();

	return 0;
}
