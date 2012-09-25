/**
 * \file PVFileDescription.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVRUSH_PVFILEDESCRIPTION_H
#define PVRUSH_PVFILEDESCRIPTION_H

#include <pvkernel/core/PVFileSerialize.h>
#include <pvkernel/rush/PVInputDescription.h>

#include <pvkernel/core/PVRecentItemsManager.h>

namespace PVRush {

class PVFileDescription: public PVInputDescription
{
	friend class PVCore::PVSerializeObject;
public:
	PVFileDescription(QString const& path):
		_path(path),
		_was_serialized(false)
	{ set_path(path); }

public:
	PVFileDescription():
		_was_serialized(false)
	{ };

public:
	operator QString() const { return _path; }
	operator QString&() { return _path; }
	operator QString const& () const { return _path; }

public:
	virtual bool operator==(const PVInputDescription& other) const
	{
		return _path == ((PVFileDescription&)other)._path;
	}

public:
	// For historical reason
	QString toString() const { return _path; }

	QString human_name() const { return _path; }
	QString path() const { return _path; }

public:
	virtual void save_to_qsettings(QSettings& settings) const
	{
		settings.setValue("path", path());
	}

	virtual void load_from_qsettings(const QSettings& settings)
	{
		_path = settings.value("path").toString();
	}

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
		PVCore::PVFileSerialize fs(_path);
		if (so.object("original", fs, "Include original file", !_was_serialized, (PVCore::PVFileSerialize*) NULL, !_was_serialized, false)) {
			_path = fs.get_path();
			if (!so.is_writing()) {
				_was_serialized = true;
			}
		}
		else
		if (!so.is_writing()) {
			if (!QFileInfo(_path).isReadable()) {
				boost::shared_ptr<PVCore::PVSerializeArchiveError> exc(new PVCore::PVSerializeArchiveErrorFileNotReadable(_path));
				boost::shared_ptr<PVCore::PVSerializeArchiveFixAttribute> error(new PVCore::PVSerializeArchiveFixAttribute(so, exc, "file_path"));
				so.repairable_error(error);
			}
		}
	}

protected:
	QString _path;
	bool _was_serialized;
};

}

#endif
