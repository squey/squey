//
// MIT License
//
// © Squey, 2024
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

#include "PVInputTypeParquet.h"

#include <pvbase/general.h>

#include <pvkernel/core/PVArchive.h>
#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include "../../common/parquet/PVParquetAPI.h"
#include "../../common/parquet/PVParquetFileDescription.h"

#include <QMessageBox>
#include <QFileInfo>
#include <QStandardPaths>

#include <sys/time.h>

PVRush::PVInputTypeParquet::PVInputTypeParquet() : PVInputTypeDesc<PVParquetFileDescription>()
{
}

bool PVRush::PVInputTypeParquet::createWidget(hash_formats& formats,
                                              list_inputs& inputs,
                                              QString& format_str,
                                              PVCore::PVArgumentList& /*args_ext*/,
                                              QWidget* parent) const
{
	// Get information from file dialog
	QStringList file_paths = PVWidgets::PVFileDialog::getOpenFileNames(
	    parent,
		tr("Import Apache Parquet files"),
		QStandardPaths::writableLocation(QStandardPaths::HomeLocation),
		tr("Apache Parquet files (*.parquet)")
	);
	if (file_paths.size() == 0) {
		return false;
	}

	PVFormat format;
	if (create_source_description_params(file_paths, inputs, format)) {
		formats[QString("custom")] = format;
		format_str = "custom";
		return true;
	}
	else {
		return false;
	}
}

bool PVRush::PVInputTypeParquet::create_source_description_params(const QString& params_json, list_inputs& inputs, PVFormat& format) const
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
		QMessageBox::critical(nullptr, tr("Parquet | Import data parameters error"), "Unable to import data using the following JSON parameters: \n\n" + params_json);
		return false;
	}
	return create_source_description_params(paths, inputs, format);
}

bool PVRush::PVInputTypeParquet::create_source_description_params(const QStringList& paths, list_inputs& inputs, PVFormat& format) const
{
	try {
		PVParquetFileDescription* input_desc = new PVParquetFileDescription(paths);
		PVParquetAPI api(input_desc);

		if (not api.same_schemas()) {
			QMessageBox::critical(nullptr, tr("Incompatible schemas"), tr("All files should have the exact same schema in order to be loaded as a concatenation."));
			return false;
		}

		format = PVRush::PVFormat(api.get_format().documentElement());
		inputs.push_back(PVInputDescription_p(input_desc));
	}
	catch (const std::exception& e) {
		QMessageBox::critical(nullptr, tr("Error when loading files"), e.what());
		return false;
	}

	return true;
}

PVWidgets::PVExporterWidgetInterface*
PVRush::PVInputTypeParquet::create_exporter_widget(const list_inputs& inputs,
                                                   PVRush::PVNraw const& nraw) const
{
	return new PVWidgets::PVParquetExporterWidget(inputs, nraw);
}

std::unique_ptr<PVRush::PVExporterBase>
PVRush::PVInputTypeParquet::create_exporter(const list_inputs& inputs,
                                         PVRush::PVNraw const& nraw) const
{
	return std::make_unique<PVRush::PVParquetExporter>(inputs, nraw);
}

QString PVRush::PVInputTypeParquet::get_exporter_filter_string(const list_inputs& /*inputs*/) const
{
	return ".parquet files (*.parquet)";
}

QString PVRush::PVInputTypeParquet::name() const
{
	return {"parquet"};
}

QString PVRush::PVInputTypeParquet::human_name() const
{
	return {"Parquet import plugin"};
}

QString PVRush::PVInputTypeParquet::human_name_serialize() const
{
	return QString(tr("Parquet"));
}

QString PVRush::PVInputTypeParquet::internal_name() const
{
	return {"01-parquet"};
}

QString PVRush::PVInputTypeParquet::menu_input_name() const
{
	return {"Parquet"};
}

QString PVRush::PVInputTypeParquet::tab_name_of_inputs(list_inputs const& in) const
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
