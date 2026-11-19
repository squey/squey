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
#include <pvkernel/widgets/PVImportFilterDialog.h>

#include <rapidjson/document.h>

#include <QFileInfo>
#include <QMessageBox>
#include <QStandardPaths>

#include <cassert>
#include <optional>
#include <string>
#include <vector>

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
    QWidget* parent) const
{
	PVFormat format;
	list_inputs described;
	if (describe_files(input_paths, described, format) and choose(described, format, parent)) {
		inputs << described;
		formats[QString("custom")] = format;
		format_str = "custom";
		return true;
	} else {
		return false;
	}
}

bool PVRush::PVInputTypeFileList::choose(const list_inputs& inputs,
                                         PVFormat& format,
                                         QWidget* parent) const
{
	auto& input_desc = static_cast<PVFileListDescription&>(*inputs.front());
	const QHash<QString, double> rates = fill_rates(input_desc);
	std::vector<PVWidgets::PVImportFilterDialog::Column> columns;
	const QList<PVAxisFormat>& axes = format.get_axes();
	// But for the column of the names of the files, which a source of several files always has.
	for (qsizetype axis = input_desc.multi_inputs() ? 1 : 0; axis < axes.size(); axis++) {
		const QString& name = axes[axis].get_name();
		columns.push_back(
		    {name, axes[axis].get_type(),
		     rates.contains(name) ? std::optional<double>(rates[name]) : std::nullopt});
	}

	PVWidgets::PVImportFilterDialog dialog(
	    columns, input_desc.columns(), input_desc.row_filter(),
	    [this, &input_desc](const QString& condition) {
		    return row_filter_error(input_desc, condition);
	    },
	    parent);
	if (dialog.exec() != QDialog::Accepted) {
		return false;
	}
	input_desc.set_columns(dialog.columns());
	input_desc.set_row_filter(dialog.row_filter());
	format = get_format_from_inputs(inputs);
	return format.get_axes().size() > 0;
}

bool PVRush::PVInputTypeFileList::create_source_description_params(const QString& params_json,
                                                                   list_inputs& inputs,
                                                                   PVFormat& format) const
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
		QMessageBox::critical(
		    nullptr, tr("%1 | Import data parameters error").arg(menu_input_name()),
		    "Unable to import data using the following JSON parameters: \n\n" + params_json);
		return false;
	}

	list_inputs described;
	if (not describe_files(paths, described, format)) {
		return false;
	}
	auto& input_desc = static_cast<PVFileListDescription&>(*described.front());
	if (not columns.isEmpty() or not row_filter.isEmpty()) {
		if (const QString error =
		        row_filter.isEmpty() ? QString() : row_filter_error(input_desc, row_filter);
		    not error.isEmpty()) {
			QMessageBox::critical(nullptr, tr("%1 | Rows to import").arg(menu_input_name()), error);
			return false;
		}
		input_desc.set_columns(columns);
		input_desc.set_row_filter(row_filter);
		format = get_format_from_inputs(described);
	}
	inputs << described;
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
