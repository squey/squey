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
		return QString::fromStdString(_source_names[_current_source_index - 1]);
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
		std::vector<std::tuple<std::string /* pointer */, std::string /* pointer_base */, std::string /* source_name */>> pointers_source = {
		    {"/post/constant/connectivities", "", "connectivities"},
			{"/post/constant/entityresults", "", "constant"},
			{"/post/singlestate/entityresults", "/post/singlestate/states", "results"}
		};

		size_t index = 0;
		for (auto& [pointer_source, pointer_base, source_name] : pointers_source) {
			const rapidjson::Value* source_node =
			    rapidjson::Pointer(pointer_source.c_str()).Get(_selected_nodes);
			if (source_node) {
				std::vector<std::string> entity_types;
				if(source_node->IsObject()) {
					for (const auto& entity_type : source_node->GetObject()) {
						entity_types.emplace_back(entity_type.name.GetString());
						rapidjson::Document doc;
						if (not pointer_base.empty()) {
							const rapidjson::Value* base_node =
								rapidjson::Pointer(pointer_base.c_str()).Get(_selected_nodes);
							assert(base_node);
							rapidjson::Pointer(pointer_base.c_str()).Set(doc, *base_node);
						}
						const std::string& subpointer_source = pointer_source + "/" + entity_type.name.GetString();
						rapidjson::Pointer(subpointer_source.c_str()).Set(doc, entity_type.value);
						_selected_nodes_by_source.emplace_back(std::move(doc));
					}
				}
				else {
					rapidjson::Document doc;
					rapidjson::Pointer(pointer_source.c_str()).Set(doc, *source_node);
 					_selected_nodes_by_source.emplace_back(std::move(doc));
				}

				if (entity_types.size() > 1) {
					for (const std::string& entity_type : entity_types) {
						_source_names.emplace_back(source_name + "/" + entity_type);
					}
				}
				else {
					_source_names.emplace_back(source_name);
				}

			}
		}

#if 0
		for (const rapidjson::Document& doc : _selected_nodes_by_source) {
			rapidjson::StringBuffer buffer;
			buffer.Clear();
			rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
			doc.Accept(writer);
			pvlogger::fatal() << buffer.GetString() << std::endl;
		}
#endif
	}

  private:
	rapidjson::Document _selected_nodes;
	std::vector<rapidjson::Document> _selected_nodes_by_source;
	std::vector<std::string> _source_names;
	mutable size_t _current_source_index = 0;
};

} // namespace PVRush

#endif // __PVPCAPDESCRIPTION_H__
