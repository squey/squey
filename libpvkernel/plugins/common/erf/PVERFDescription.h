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
	PVERFDescription(QString const& path, rapidjson::Document&& selected_nodes)
	    : PVFileDescription(path, false /*multi_inputs*/)
	    , _selected_nodes(std::move(selected_nodes))
	{
		split_selected_nodes_by_sources(_selected_nodes);
	}

  public: // FIXME : not working properly
	QString human_name() const override
	{
		std::vector<std::string> sources{"connectivities", "constant", "results"};
		return QString::fromStdString(sources[_sources_index[_current_source_index - 1]]);
	}

  public:
	rapidjson::Document& selected_nodes() { return _selected_nodes; }

  public:
	const rapidjson::Document& current_source_selected_nodes()
	{
		return _selected_nodes_by_source[_current_source_index++];
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
		} else {
			so.attribute_write("file_path", _path);
		}

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

		// Deserialize nodes selection
		rapidjson::Document selected_nodes;
		selected_nodes.Parse<0>(so.attribute_read<QString>("selected_nodes").toStdString().c_str());

		return std::unique_ptr<PVInputDescription>(
		    new PVERFDescription(path, std::move(selected_nodes)));
	}

  private:
	void split_selected_nodes_by_sources(const rapidjson::Document& selected_nodes)
	{
		std::vector<std::string> pointers_source = {
		    "/post/constant/connectivities", "/post/constant/entityresults", "/post/singlestate"};

		for (size_t i = 0; i < pointers_source.size(); i++) {
			const std::string& pointer_source = pointers_source[i];
			const rapidjson::Value* source_node =
			    rapidjson::Pointer(pointer_source.c_str()).Get(_selected_nodes);
			if (source_node) {
				rapidjson::Document doc;
				rapidjson::Pointer(pointer_source.c_str()).Set(doc, *source_node);
				_selected_nodes_by_source.emplace_back(std::move(doc));

				_sources_index.emplace_back(i);
			}
		}
	}

  private:
	rapidjson::Document _selected_nodes;
	std::vector<rapidjson::Document> _selected_nodes_by_source;
	std::vector<size_t> _sources_index;
	size_t _current_source_index = 0;
};

} // namespace PVRush

#endif // __PVPCAPDESCRIPTION_H__
