/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2017
 */
#ifndef __PVPCAPDESCRIPTION_H__
#define __PVPCAPDESCRIPTION_H__

#include <pvkernel/rush/PVFileDescription.h>

namespace PVRush
{

class PVERFDescription : public PVFileDescription
{
  public:
	PVERFDescription(QString const& path) : PVFileDescription(path, false /*multi_inputs*/) {}

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
		} else {
			so.attribute_write("file_path", _path);
		}
	}

	static std::unique_ptr<PVInputDescription> serialize_read(PVCore::PVSerializeObject& so)
	{
		so.set_current_status("Loading source file information...");
		QString path = so.attribute_read<QString>("file_path");

		// File exists, continue with it
		if (QFileInfo(path).isReadable()) {
			// FIXME : As long as the pcap preprocessing phase is not properly integrated to
			//         the import pipeline, it will not be possible to reload an investigation
			//         with a missing cache (ie. from the original pcap files)
			return std::unique_ptr<PVInputDescription>(new PVERFDescription(path));
		} else {
			throw PVCore::PVSerializeReparaibleFileError(
			    "Source file: '" + path.toStdString() + "' can't be found",
			    so.get_logical_path().toStdString(), path.toStdString());
		}
	}

  private:
};

} // namespace PVRush

#endif // __PVPCAPDESCRIPTION_H__
