/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVFILEDESCRIPTION_H
#define PVRUSH_PVFILEDESCRIPTION_H

#include <pvkernel/core/PVFileSerialize.h>
#include <pvkernel/rush/PVInputDescription.h>

#include <pvkernel/core/PVRecentItemsManager.h>

namespace PVRush
{

class PVFileDescription : public PVInputDescription
{
	friend class PVCore::PVSerializeObject;

  public:
	PVFileDescription(QString const& path) : _path(QDir().absoluteFilePath(path)) {}

  public:
	virtual bool operator==(const PVInputDescription& other) const
	{
		return _path == ((PVFileDescription&)other)._path;
	}

  public:
	QString human_name() const { return _path; }
	QString path() const { return _path; }

  public:
	virtual void save_to_qsettings(QSettings& settings) const { settings.setValue("path", path()); }

	static std::unique_ptr<PVRush::PVInputDescription>
	load_from_qsettings(const QSettings& settings)
	{
		return std::unique_ptr<PVFileDescription>(
		    new PVFileDescription(settings.value("path").toString()));
	}

  public:
	void serialize_write(PVCore::PVSerializeObject& so)
	{
		so.attribute("file_path", _path);
		PVCore::PVFileSerialize fs(_path);
		so.object("original", fs, "Include original file", true, (PVCore::PVFileSerialize*)nullptr,
		          true, false);
	}

	static std::unique_ptr<PVInputDescription> serialize_read(PVCore::PVSerializeObject& so)
	{
		QString path;
		so.attribute("file_path", path);

		PVCore::PVFileSerialize fs(path);
		if (so.object("original", fs, "Include original file", true,
		              (PVCore::PVFileSerialize*)nullptr, true, false)) {
			path = fs.get_path();
		}

		if (not QFileInfo(path).isReadable()) {
			std::shared_ptr<PVCore::PVSerializeArchiveError> exc(
			    new PVCore::PVSerializeArchiveErrorFileNotReadable(path.toStdString()));
			std::shared_ptr<PVCore::PVSerializeArchiveFixAttribute> error(
			    new PVCore::PVSerializeArchiveFixAttribute(so, exc, "file_path"));
			so.repairable_error(error);
			return nullptr;
		}

		return std::unique_ptr<PVInputDescription>(new PVFileDescription(path));
	}

  protected:
	QString _path;
};
}

#endif
