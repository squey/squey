/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
	PVERFDescription(QStringList const& paths,
	                 QString const& name,
	                 rapidjson::Document&& selected_nodes)
	    : PVFileDescription(paths.front(), paths.size() > 1 /*multi_inputs*/)
	    , _name(name)
	    , _selected_nodes(std::move(selected_nodes))
	    , _paths(paths)
	{
		rapidjson::StringBuffer buffer;
		buffer.Clear();
		rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
		_selected_nodes.Accept(writer);
		pvlogger::info() << buffer.GetString() << std::endl;
	}

  public:
	QString human_name() const override { return _name; }

	QStringList paths() const { return _paths; }

  public:
	rapidjson::Document& selected_nodes() { return _selected_nodes; }

  public:
	void serialize_write(PVCore::PVSerializeObject& so) const override
	{
		so.set_current_status("Saving source file information...");

		if (so.save_log_file()) {
			for (const QString& path : _paths) {
				QFileInfo fi(path.front());
				QString fname = fi.fileName();
				so.file_write(fname, path);
				// FIXME : Before, we still reload data from original file, never from packed file.
				so.attribute_write("file_path", fname);
			}
		} else {
			so.attribute_write("files_path", _paths);
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
		QStringList paths = so.attribute_read<QStringList>("files_path");

		// File exists, continue with it
		for (const QString& path : paths) {
			if (not QFileInfo(path).isReadable()) {
				throw PVCore::PVSerializeReparaibleFileError(
				    "Source file: '" + path.toStdString() + "' can't be found",
				    so.get_logical_path().toStdString(), path.toStdString());
			}
		}

		QString name = so.attribute_read<QString>("name");

		// Deserialize nodes selection
		rapidjson::Document selected_nodes;
		selected_nodes.Parse<0>(so.attribute_read<QString>("selected_nodes").toStdString().c_str());

		return std::unique_ptr<PVInputDescription>(
		    new PVERFDescription(paths, name, std::move(selected_nodes)));
	}

  private:
	QString _name;
	rapidjson::Document _selected_nodes;
	QStringList _paths;
};

} // namespace PVRush

#endif // __PVPCAPDESCRIPTION_H__
