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

#include <pvkernel/widgets/PVFileDialog.h>

#include <rapidjson/document.h>

#include <QFileInfo>
#include <QMessageBox>
#include <QStandardPaths>

#include <cassert>
#include <string>

bool PVRush::PVInputTypeFileList::create_widget(hash_formats& formats,
                                                list_inputs& inputs,
                                                QString& format_str,
                                                PVCore::PVArgumentList& args_ext,
                                                QWidget* parent) const
{
	// Get information from file dialog
	QStringList file_paths = PVWidgets::PVFileDialog::getOpenFileNames(
	    parent, files_dialog_title(),
	    QStandardPaths::writableLocation(QStandardPaths::HomeLocation), files_dialog_filter());
	if (file_paths.size() == 0) {
		return false;
	}

	return create_widget_with_input_files(file_paths, formats, inputs, format_str, args_ext,
	                                      parent);
}

bool PVRush::PVInputTypeFileList::create_widget_with_input_files(
    const QStringList& input_paths,
    hash_formats& formats,
    list_inputs& inputs,
    QString& format_str,
    PVCore::PVArgumentList& /*args_ext*/,
    QWidget* /*parent*/) const
{
	PVFormat format;
	if (describe_files(input_paths, inputs, format)) {
		formats[QString("custom")] = format;
		format_str = "custom";
		return true;
	} else {
		return false;
	}
}

bool PVRush::PVInputTypeFileList::create_source_description_params(const QString& params_json,
                                                                   list_inputs& inputs,
                                                                   PVFormat& format) const
{
	QStringList paths;
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
	}
	if (paths.isEmpty()) {
		QMessageBox::critical(
		    nullptr, tr("%1 | Import data parameters error").arg(menu_input_name()),
		    "Unable to import data using the following JSON parameters: \n\n" + params_json);
		return false;
	}
	return describe_files(paths, inputs, format);
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
