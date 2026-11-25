//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/rush/PVInputTypeFileList.h>

#include <pvkernel/core/PVTaskProgress.h>
#include <pvkernel/widgets/PVFileListImportWidget.h>

#include <rapidjson/document.h>

#include <QFileInfo>

#include <cassert>
#include <stdexcept>
#include <string>

PVWidgets::PVImportWidget* PVRush::PVInputTypeFileList::create_import_widget(QWidget* parent) const
{
	return new PVWidgets::PVFileListImportWidget(*this, parent);
}

bool PVRush::PVInputTypeFileList::create_source_description_params(
    const QString& params_json,
    list_inputs& inputs,
    PVFormat& format,
    PVCore::PVTaskProgress& progress) const
{
	QStringList paths;
	QStringList columns;
	QString row_filter;
	rapidjson::Document json;
	json.Parse<0>(params_json.toStdString().c_str());
	if (not json.HasParseError()) {
		if (json.HasMember("paths")) {
			const rapidjson::Value& json_paths = json["paths"];
			if (json_paths.IsArray()) {
				for (rapidjson::SizeType i = 0; i < json_paths.Size(); i++) {
					const std::string& path_str = json_paths[i].GetString();
					paths << QString::fromStdString(path_str);
				}
			}
		}
		if (json.HasMember("columns") and json["columns"].IsArray()) {
			for (const rapidjson::Value& column : json["columns"].GetArray()) {
				if (column.IsString()) {
					columns << QString::fromUtf8(column.GetString(), column.GetStringLength());
				}
			}
		}
		if (json.HasMember("row_filter") and json["row_filter"].IsString()) {
			row_filter = QString::fromUtf8(json["row_filter"].GetString(),
			                               json["row_filter"].GetStringLength());
		}
	}
	if (paths.isEmpty()) {
		throw std::runtime_error(
		    tr("Unable to import data using the following JSON parameters:\n\n%1")
		        .arg(params_json)
		        .toStdString());
	}

	QStringList warnings;
	std::shared_ptr<PVFileListDescription> input_desc = describe(paths, progress, warnings);
	if (input_desc == nullptr) {
		return false;
	}
	if (not row_filter.isEmpty()) {
		if (const QString error = row_filter_error(*input_desc, row_filter); not error.isEmpty()) {
			throw std::runtime_error(error.toStdString());
		}
	}
	input_desc->set_columns(columns);
	input_desc->set_row_filter(row_filter);
	inputs << input_desc;
	format = get_format_from_inputs(inputs);
	return true;
}

QString PVRush::PVInputTypeFileList::tab_name_of_inputs(list_inputs const& in) const
{
	QString tab_name;
	auto* f = dynamic_cast<PVFileDescription*>(in[0].get());
	assert(f);
	QFileInfo fi(f->path());
	if (in.count() == 1) {
		tab_name = fi.fileName();
	} else {
		tab_name = fi.canonicalPath();
	}
	return tab_name;
}
