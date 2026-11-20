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

#include "PVInputTypeJson.h"

#include "../../common/arrow/PVArrowRowFilter.h"
#include "../../common/json/PVJsonFileDescription.h"
#include "../../common/json/PVJsonLines.h"
#include "../../common/json/PVJsonSchema.h"
#include "../../common/parquet/PVParquetAPI.h"

#include <pvkernel/core/PVTaskProgress.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr const char* EXTENSIONS[] = {"json", "jsonl", "ndjson"};

} // namespace

PVRush::PVInputTypeJson::PVInputTypeJson() : PVInputTypeFileList() {}

QString PVRush::PVInputTypeJson::files_dialog_title() const
{
	return tr("Import JSON files");
}

QString PVRush::PVInputTypeJson::files_dialog_filter() const
{
	// Followed by the extension of a compression, or by the number or the date of a rotation.
	QStringList patterns;
	for (const QString extension : EXTENSIONS) {
		patterns << "*." + extension << "*." + extension + ".*" << "*." + extension + "-*";
	}
	return tr("JSON files of one object per line (%1)").arg(patterns.join(' '));
}

std::shared_ptr<const PVRush::PVJsonSchema> PVRush::PVInputTypeJson::schema_of(
    const QStringList& paths, PVCore::PVTaskProgress* progress, QStringList* warnings)
{
	std::shared_ptr<const PVJsonSchema> schema =
	    PVJsonSchema::infer(paths, [progress](size_t bytes_read, size_t size) {
		    if (progress == nullptr) {
			    return true;
		    }
		    progress->set_maximum(size);
		    progress->set_value(bytes_read);
		    return not progress->canceled();
	    });
	if (schema == nullptr) {
		return nullptr;
	}

	if (schema->columns().empty()) {
		throw std::runtime_error(
		    (schema->row_count() == 0
		         ? tr("These files hold no JSON object on a line of its own: Squey reads files of "
		              "one JSON object per line, such as JSON Lines.")
		         : tr("The JSON objects of these files have no field with a value."))
		        .toStdString());
	}
	if (warnings != nullptr and schema->invalid_line_count() > 0) {
		*warnings << tr("Lines of these files that are not JSON objects are left out: %1 of them, "
		                "the first one being line %2 of \"%3\".")
		                 .arg(schema->invalid_line_count())
		                 .arg(schema->first_invalid_line())
		                 .arg(schema->first_invalid_file());
	}
	return schema;
}

std::shared_ptr<PVRush::PVFileListDescription> PVRush::PVInputTypeJson::describe(
    const QStringList& paths, PVCore::PVTaskProgress& progress, QStringList& warnings) const
{
	progress.set_status(tr("Reading the fields of the JSON objects"));
	std::shared_ptr<const PVJsonSchema> schema = schema_of(paths, &progress, &warnings);
	if (schema == nullptr) {
		return nullptr;
	}
	auto input_desc = std::make_shared<PVJsonFileDescription>(paths);
	input_desc->set_schema(std::move(schema));
	return input_desc;
}

std::shared_ptr<PVRush::PVFileListDescription>
PVRush::PVInputTypeJson::with_paths(const PVFileListDescription& description,
                                    const QStringList& paths) const
{
	auto input_desc = std::make_shared<PVJsonFileDescription>(paths);
	input_desc->set_schema(static_cast<const PVJsonFileDescription&>(description).schema());
	return input_desc;
}

QHash<QString, double>
PVRush::PVInputTypeJson::fill_rates(const PVFileListDescription& input_desc) const
{
	QHash<QString, double> rates;
	const PVJsonSchema& schema = *static_cast<const PVJsonFileDescription&>(input_desc).schema();
	for (const PVJsonSchema::Column& column : schema.columns()) {
		rates[QString::fromStdString(column.name)] =
		    schema.row_count() == 0 ? 0. : double(column.value_count) / double(schema.row_count());
	}
	return rates;
}

QString PVRush::PVInputTypeJson::row_filter_error(const PVFileListDescription& input_desc,
                                                  const QString& condition) const
{
	const auto& json_desc = static_cast<const PVJsonFileDescription&>(input_desc);
	return QString::fromStdString(PVArrowRowFilter::error_of(
	    condition.toStdString(), json_desc.schema()->arrow_schema(json_desc.multi_inputs())));
}

PVRush::PVFormat
PVRush::PVInputTypeJson::get_format_from_inputs(list_inputs const& inputs,
                                                PVCore::PVTaskProgress* progress) const
{
	auto* input_desc =
	    inputs.isEmpty() ? nullptr : dynamic_cast<PVJsonFileDescription*>(inputs.front().get());
	if (input_desc == nullptr) {
		return {};
	}

	if (input_desc->schema() == nullptr) {
		std::shared_ptr<const PVJsonSchema> schema =
		    schema_of(input_desc->paths(), progress, nullptr);
		if (schema == nullptr) {
			return {};
		}
		input_desc->set_schema(std::move(schema));
	}

	std::vector<size_t> column_indexes;
	return PVFormat(PVParquetAPI::format_of(
	                    input_desc->schema()->with_columns(input_desc->columns())->arrow_schema(),
	                    input_desc->multi_inputs(), column_indexes)
	                    .documentElement());
}

PVRush::PVInputDescription_p PVRush::PVInputTypeJson::serialize_read(PVCore::PVSerializeObject& so)
{
	auto input_desc =
	    std::make_shared<PVJsonFileDescription>(PVFileListDescription::read_paths(so));
	input_desc->read_filters(so);
	return input_desc;
}

PVRush::PVInputDescription_p
PVRush::PVInputTypeJson::load_input_from_string(pvcop::db::format const& fields,
                                                bool /*multi_inputs*/)
{
	auto input_desc =
	    std::make_shared<PVJsonFileDescription>(PVFileListDescription::paths_of(fields));
	input_desc->read_filters(fields);
	return input_desc;
}

QStringList PVRush::PVInputTypeJson::get_supported_extensions() const
{
	const QStringList compressions = PVJsonLines::compressions();
	QStringList extensions;
	for (const QString extension : EXTENSIONS) {
		extensions << extension;
		for (const QString& compression : compressions) {
			extensions << extension + "." + compression;
		}
	}
	return extensions;
}

QString PVRush::PVInputTypeJson::name() const
{
	return {"json"};
}

QString PVRush::PVInputTypeJson::human_name() const
{
	return {"JSON import plugin"};
}

QString PVRush::PVInputTypeJson::human_name_serialize() const
{
	return QString(tr("JSON"));
}

QString PVRush::PVInputTypeJson::internal_name() const
{
	return {"05-json"};
}

QString PVRush::PVInputTypeJson::menu_input_name() const
{
	return {"JSON"};
}
