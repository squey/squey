/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVFILEDESCRIPTION_H
#define PVRUSH_PVFILEDESCRIPTION_H

#include <pvkernel/rush/PVInputDescription.h>

#include <pvcop/db/format.h>

#include <QFile>

namespace PVRush
{

class PVFileDescription : public PVInputDescription
{
	friend class PVCore::PVSerializeObject;

  public:
	PVFileDescription(QString const& path, bool multi_inputs = false)
	    : _path(QDir().absoluteFilePath(path)), _multi_inputs(multi_inputs)
	{
		if (not QFile::exists(_path)) {
			throw PVRush::BadInputDescription("Input file '" + _path.toStdString() +
			                                  "' doesn't exists");
		}
	}

  public:
	bool operator==(const PVInputDescription& other) const override
	{
		return _path == ((PVFileDescription&)other)._path;
	}

  public:
	QString human_name() const override { return _path; }
	QString path() const { return _path; }
	bool multi_inputs() const { return _multi_inputs; }

  public:
	void save_to_qsettings(QSettings& settings) const override
	{
		settings.setValue("path", path());
	}

	static std::unique_ptr<PVRush::PVInputDescription>
	load_from_string(pvcop::db::format const& format, bool multi_inputs)
	{
		return std::unique_ptr<PVFileDescription>(
		    new PVFileDescription(QString::fromStdString(format[0]), multi_inputs));
	}

	static std::vector<std::string> desc_from_qsetting(QSettings const& s)
	{
		return {1, s.value("path").toString().toStdString()};
	}

  public:
	void serialize_write(PVCore::PVSerializeObject& so) const override
	{
		so.set_current_status("Saving source file information...");

		if (so.save_log_file()) {
			QFileInfo fi(_path);
			QString fname = fi.fileName();
			so.file_write(fname, _path);
			// FIXME : Before, we still reload data from original file, never from packed file.
			so.attribute_write("file_path", fname);
			so.attribute_write("multi_inputs", _multi_inputs);
		} else {
			so.attribute_write("file_path", _path);
			so.attribute_write("multi_inputs", _multi_inputs);
		}
	}

	static std::unique_ptr<PVInputDescription> serialize_read(PVCore::PVSerializeObject& so)
	{
		so.set_current_status("Loading source file information...");
		QString path = so.attribute_read<QString>("file_path");
		bool multi_inputs = so.attribute_read<bool>("multi_inputs");

		// File exists, continue with it
		if (QFileInfo(path).isReadable()) {
			return std::unique_ptr<PVInputDescription>(new PVFileDescription(path, multi_inputs));
		}

		// File doesn't exists. Look for packed file
		try {
			path = so.file_read(path);
		} catch (PVCore::PVSerializeArchiveError const&) {

			// Otherwise, ask where it is.
			if (so.is_repaired_error()) {
				so.set_current_status("Original source file not found, asking for its location...");
				path = QString::fromStdString(so.get_repaired_value());
			} else {
				throw PVCore::PVSerializeReparaibleFileError(
				    "Source file: '" + path.toStdString() + "' can't be found",
				    so.get_logical_path().toStdString(), path.toStdString());
			}
		}

		return std::unique_ptr<PVInputDescription>(new PVFileDescription(path, multi_inputs));
	}

  protected:
	QString _path;
	bool _multi_inputs;
};
} // namespace PVRush

#endif
