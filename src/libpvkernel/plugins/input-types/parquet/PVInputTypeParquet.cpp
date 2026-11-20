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
#include "../../common/arrow/PVArrowRowFilter.h"

#include <parquet/file_reader.h>
#include <parquet/metadata.h>
#include <parquet/statistics.h>

#include <QSet>

#include <QFileInfo>
#include <QStandardPaths>

#include <sys/time.h>

#include <memory>
#include <stdexcept>

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

std::shared_ptr<PVRush::PVFileListDescription>
PVRush::PVInputTypeParquet::describe(const QStringList& paths,
                                     PVCore::PVTaskProgress& /*progress*/,
                                     QStringList& /*warnings*/) const
{
	auto input_desc = std::make_shared<PVFileListDescription>(paths);
	if (not PVParquetAPI(input_desc.get()).same_schemas()) {
		throw std::runtime_error(tr("All files should have the exact same schema in order to be "
		                            "loaded as a concatenation.")
		                             .toStdString());
	}
	return input_desc;
}

std::shared_ptr<PVRush::PVFileListDescription>
PVRush::PVInputTypeParquet::with_paths(const PVFileListDescription& /*description*/,
                                       const QStringList& paths) const
{
	return std::make_shared<PVFileListDescription>(paths);
}

QHash<QString, double>
PVRush::PVInputTypeParquet::fill_rates(const PVFileListDescription& input_desc) const
{
	QHash<QString, int64_t> nulls;
	QSet<QString> unknown;
	int64_t row_count = 0;
	try {
		for (const QString& path : input_desc.paths()) {
			const std::shared_ptr<parquet::FileMetaData> metadata =
			    parquet::ParquetFileReader::OpenFile(path.toStdString())->metadata();
			row_count += metadata->num_rows();
			const parquet::SchemaDescriptor* schema = metadata->schema();
			for (int leaf = 0; leaf < schema->num_columns(); leaf++) {
				const parquet::ColumnDescriptor* column = schema->Column(leaf);
				if (column->max_repetition_level() > 0) {
					continue;
				}
				// The name of a field of a structure is that of its flattened column.
				const QString name = QString::fromStdString(column->path()->ToDotString());
				for (int group = 0; group < metadata->num_row_groups(); group++) {
					const std::shared_ptr<parquet::Statistics> statistics =
					    metadata->RowGroup(group)->ColumnChunk(leaf)->statistics();
					if (statistics == nullptr or not statistics->HasNullCount()) {
						unknown.insert(name);
					} else {
						nulls[name] += statistics->null_count();
					}
				}
			}
		}
	} catch (const std::exception&) {
		return {};
	}

	QHash<QString, double> rates;
	for (auto column = nulls.cbegin(); column != nulls.cend(); ++column) {
		if (row_count > 0 and not unknown.contains(column.key())) {
			rates[column.key()] = 1. - double(column.value()) / double(row_count);
		}
	}
	return rates;
}

QString PVRush::PVInputTypeParquet::row_filter_error(const PVFileListDescription& input_desc,
                                                     const QString& condition) const
{
	try {
		PVParquetAPI api(&input_desc);
		std::shared_ptr<arrow::Schema> schema;
		const arrow::Status status = api.arrow_reader()->GetSchema(&schema);
		if (not status.ok()) {
			return QString::fromStdString(status.ToString());
		}
		// The columns the source hands the condition.
		const std::shared_ptr<arrow::Schema> flat_schema = PVParquetAPI::flatten_schema(schema);
		arrow::FieldVector fields;
		if (api.multi_inputs()) {
			fields.push_back(arrow::field(PVParquetAPI::input_column_name, arrow::utf8()));
		}
		for (int column : PVParquetAPI::row_filter_columns(*flat_schema)) {
			fields.push_back(flat_schema->field(column));
		}
		return QString::fromStdString(
		    PVArrowRowFilter::error_of(condition.toStdString(), arrow::schema(fields)));
	} catch (const std::exception& e) {
		return e.what();
	}
}

PVRush::PVFormat
PVRush::PVInputTypeParquet::get_format_from_inputs(list_inputs const& inputs,
                                                   PVCore::PVTaskProgress* /*progress*/) const
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
