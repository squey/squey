/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVFileSerialize.h>
#include <pvkernel/core/PVSerializeObject.h>

#include <QFileInfo>

PVCore::PVFileSerialize::PVFileSerialize(QString const& path) : _path(path)
{
}

QString const& PVCore::PVFileSerialize::get_path() const
{
	return _path;
}

void PVCore::PVFileSerialize::serialize_read(PVCore::PVSerializeObject& so)
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
