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
#include <pvkernel/rush/PVFileListDescription.h>
#include "PVParquetNrawExporter.h"

#include <QMessageBox>
#include <QFileInfo>
#include <QStandardPaths>

#include <sys/time.h>

PVRush::PVInputTypeParquet::PVInputTypeParquet() : PVInputTypeFileList()
{
}

QString PVRush::PVInputTypeParquet::files_dialog_title() const
{
	return tr("Import Apache Parquet files");
}

QString PVRush::PVInputTypeParquet::files_dialog_filter() const
{
	return tr("Apache Parquet files (*.parquet)");
}

bool PVRush::PVInputTypeParquet::describe_files(const QStringList& paths, list_inputs& inputs, PVFormat& format) const
{
	try {
		PVFileListDescription* input_desc = new PVFileListDescription(paths);
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

PVRush::PVFormat PVRush::PVInputTypeParquet::get_format_from_inputs(list_inputs const& inputs) const
{
	if (inputs.isEmpty()) {
		return {};
	}

	auto* input_desc = dynamic_cast<PVFileListDescription*>(inputs.front().get());
	if (input_desc == nullptr) {
		return {};
	}

	PVParquetAPI api(input_desc);
	return PVRush::PVFormat(api.get_format().documentElement());
}

PVWidgets::PVExporterWidgetInterface*
PVRush::PVInputTypeParquet::create_exporter_widget(const list_inputs& inputs,
                                                   PVRush::PVNraw const& nraw) const
{
	// This exporter copies the rows of the files, which a condition on the rows makes others than
	// those of the NRAW: the export of the NRAW writes these instead.
	auto* input_desc = dynamic_cast<PVFileListDescription*>(inputs.front().get());
	if (input_desc != nullptr and not input_desc->row_filter().isEmpty()) {
		return nullptr;
	}

	// The codec of the input, unless it cannot be read anymore, which the export then reports.
	arrow::Compression::type codec = arrow::Compression::UNCOMPRESSED;
	try {
		PVParquetAPI api(input_desc);
		codec = api.arrow_reader()->parquet_reader()->metadata()->RowGroup(0)->ColumnChunk(0)->compression();
	} catch (...) {
	}

	return new PVWidgets::PVParquetExporterWidget(std::make_unique<PVParquetExporter>(inputs, nraw),
	                                              codec);
}

std::unique_ptr<PVRush::PVExporterBase>
PVRush::PVInputTypeParquet::create_exporter(const list_inputs& inputs,
                                         PVRush::PVNraw const& nraw) const
{
	return std::make_unique<PVRush::PVParquetExporter>(inputs, nraw);
}

QStringList PVRush::PVInputTypeParquet::get_supported_extensions() const
{
    return { "parquet" };
}

QString PVRush::PVInputTypeParquet::get_exporter_filter_string(const list_inputs& /*inputs*/) const
{
	return ".parquet files (*.parquet)";
}

PVWidgets::PVExporterWidgetInterface*
PVRush::PVInputTypeParquet::create_nraw_exporter_widget(PVRush::PVNraw const& nraw,
                                                        QStringList const& column_names) const
{
	// The default codec of most of the writers of Parquet files.
	return new PVWidgets::PVParquetExporterWidget(
	    std::make_unique<PVParquetNrawExporter>(nraw, column_names), arrow::Compression::SNAPPY);
}

QString PVRush::PVInputTypeParquet::get_nraw_exporter_filter_string() const
{
	return get_exporter_filter_string({});
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
