/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2017
 */
#ifndef __PVPCAPDESCRIPTION_H__
#define __PVPCAPDESCRIPTION_H__

#include <pvkernel/rush/PVFileDescription.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/pointer.h>

namespace PVRush
{

class PVERFDescription : public PVFileDescription
{
  public:
	PVERFDescription(QString const& path, QString const& name, rapidjson::Document&& selected_nodes)
	    : PVFileDescription(path, false /*multi_inputs*/)
	    , _name(name)
	    , _selected_nodes(std::move(selected_nodes))
	{
	}

  public: // FIXME : not working properly
	QString human_name() const override { return _name; }

  public:
	rapidjson::Document& selected_nodes() { return _selected_nodes; }

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

		so.attribute_write("name", _name);

		// Serialize nodes selection
		rapidjson::StringBuffer buffer;
		buffer.Clear();
		rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
		_selected_nodes.Accept(writer);
		so.attribute_write("selected_nodes", buffer.GetString());
	}

	static std::unique_ptr<PVInputDescription> serialize_read(PVCore::PVSerializeObject& so)
	{
		so.set_current_status("Loading source file information...");
		QString path = so.attribute_read<QString>("file_path");

		// File exists, continue with it
		if (not QFileInfo(path).isReadable()) {
			throw PVCore::PVSerializeReparaibleFileError(
			    "Source file: '" + path.toStdString() + "' can't be found",
			    so.get_logical_path().toStdString(), path.toStdString());
		}

		QString name = so.attribute_read<QString>("name");

		// Deserialize nodes selection
		rapidjson::Document selected_nodes;
		selected_nodes.Parse<0>(so.attribute_read<QString>("selected_nodes").toStdString().c_str());

		return std::unique_ptr<PVInputDescription>(
		    new PVERFDescription(path, name, std::move(selected_nodes)));
	}

  private:
	QString _name;
	rapidjson::Document _selected_nodes;
};

} // namespace PVRush

#endif // __PVPCAPDESCRIPTION_H__
