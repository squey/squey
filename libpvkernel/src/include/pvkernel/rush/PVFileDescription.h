/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVFILEDESCRIPTION_H
#define PVRUSH_PVFILEDESCRIPTION_H

#include <pvkernel/rush/PVInputDescription.h>

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
		so.set_current_status("Serialize file");

		if (so.save_log_file()) {
			QFileInfo fi(_path);
			QString fname = fi.fileName();
			so.file(fname, _path);
		}

		so.attribute("file_path", _path);
	}

	static std::unique_ptr<PVInputDescription> serialize_read(PVCore::PVSerializeObject& so)
	{
		so.set_current_status("Searching for source file.");
		QString path;
		so.attribute("file_path", path);

		if (not QFileInfo(path).isReadable()) {
			if (so.is_repaired_error()) {
				path = QString::fromStdString(so.get_repaired_value());
			} else {
				throw PVCore::PVSerializeReparaibleError("Can't find source file",
				                                         so.get_logical_path().toStdString(),
				                                         path.toStdString());
			}
		}

		return std::unique_ptr<PVInputDescription>(new PVFileDescription(path));
	}

  protected:
	QString _path;
};
}

#endif
