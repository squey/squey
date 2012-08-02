/**
 * \file PVFileSerialize.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVFileSerialize.h>


PVCore::PVFileSerialize::PVFileSerialize(QString const& path):
	_path(path)
{
}

QString const& PVCore::PVFileSerialize::get_path() const
{
	return _path;
}

void PVCore::PVFileSerialize::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	QString fname;
	so.attribute("filename", fname);
	so.file(fname, _path);
}

void PVCore::PVFileSerialize::serialize_write(PVCore::PVSerializeObject& so)
{
	// Get the file's name
	QFileInfo fi(_path);
	QString fname = fi.fileName();
	so.file(fname, _path);
	so.attribute("filename", fname);
}
