#ifndef PVRUSH_PVFILEDESCRIPTION_H
#define PVRUSH_PVFILEDESCRIPTION_H

#include <pvkernel/rush/PVInputDescription.h>

namespace PVRush {

class PVFileDescription: public PVInputDescription
{
	friend class PVCore::PVSerializeObject;
public:
	PVFileDescription(QString const& path):
		_path(path)
	{ set_path(path); }

protected:
	PVFileDescription() { };

public:
	operator QString() const { return _path; }
	operator QString&() { return _path; }
	operator QString const& () const { return _path; }

public:
	// For historical reason
	QString toString() const { return _path; }

	QString human_name() const { return _path; }
	QString path() const { return _path; }

protected:
	void set_path(QString const& path)
	{
		QDir dir;
		_path = dir.absoluteFilePath(path);
	}

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
	{
		so.attribute("file_path", _path);
	}

protected:
	QString _path;
};

}

#endif
