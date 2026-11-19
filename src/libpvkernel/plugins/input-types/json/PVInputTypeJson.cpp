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

#include "../../common/json/PVJsonFileDescription.h"
#include "../../common/json/PVJsonSchema.h"
#include "../../common/parquet/PVParquetAPI.h"

#include <pvkernel/core/PVProgressBox.h>

#include <QMessageBox>

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

PVRush::PVInputTypeJson::PVInputTypeJson() : PVInputTypeFileList() {}

QString PVRush::PVInputTypeJson::files_dialog_title() const
{
	return tr("Import JSON files");
}

QString PVRush::PVInputTypeJson::files_dialog_filter() const
{
	return tr("JSON files of one object per line (*.json *.jsonl *.ndjson)");
}

bool PVRush::PVInputTypeJson::find_schema(PVJsonFileDescription& input_desc) const
{
	constexpr const int progress_maximum = 1000;
	std::shared_ptr<const PVJsonSchema> schema;
	std::string error;
	const PVCore::PVProgressBox::CancelState state = PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {
		    pbox.set_maximum(progress_maximum);
		    // The progress box ends the program on an exception out of its thread.
		    try {
			    schema = PVJsonSchema::infer(input_desc.paths(), [&](size_t bytes_read,
			                                                         size_t size) {
				    pbox.set_value(int(progress_maximum * bytes_read / std::max<size_t>(size, 1)));
				    return pbox.get_cancel_state() == PVCore::PVProgressBox::CancelState::CONTINUE;
			    });
		    } catch (const std::exception& e) {
			    error = e.what();
		    }
	    },
	    tr("Reading the fields of the JSON objects"), nullptr);
	if (not error.empty()) {
		throw std::runtime_error(error);
	}
	if (state != PVCore::PVProgressBox::CancelState::CONTINUE or schema == nullptr) {
		return false;
	}

	if (schema->columns().empty()) {
		QMessageBox::critical(
		    nullptr, tr("No JSON values"),
		    schema->row_count() == 0
		        ? tr("These files hold no JSON object on a line of its own: Squey reads files of "
		             "one JSON object per line, such as JSON Lines.")
		        : tr("The JSON objects of these files have no field with a value."));
		return false;
	}
	if (schema->invalid_line_count() > 0) {
		QMessageBox::warning(nullptr, tr("Lines left out"),
		                     tr("Lines of these files that are not JSON objects are left out: %1 "
		                        "of them, the first one being line %2 of \"%3\".")
		                         .arg(schema->invalid_line_count())
		                         .arg(schema->first_invalid_line())
		                         .arg(schema->first_invalid_file()));
	}

	input_desc.set_schema(schema);
	return true;
}

bool PVRush::PVInputTypeJson::find_format(PVJsonFileDescription& input_desc, PVFormat& format) const
{
	if (input_desc.schema() == nullptr and not find_schema(input_desc)) {
		return false;
	}

	std::vector<size_t> column_indexes;
	format = PVFormat(PVParquetAPI::format_of(
	                      input_desc.schema()->with_columns(input_desc.columns())->arrow_schema(),
	                      input_desc.multi_inputs(), column_indexes)
	                      .documentElement());
	return true;
}

bool PVRush::PVInputTypeJson::describe_files(const QStringList& paths,
                                             list_inputs& inputs,
                                             PVFormat& format) const
{
	try {
		auto input_desc = std::make_shared<PVJsonFileDescription>(paths);
		if (not find_format(*input_desc, format)) {
			return false;
		}
		inputs.push_back(input_desc);
	} catch (const std::exception& e) {
		QMessageBox::critical(nullptr, tr("Error when loading files"), e.what());
		return false;
	}

	return true;
}

PVRush::PVFormat PVRush::PVInputTypeJson::get_format_from_inputs(list_inputs const& inputs) const
{
	auto* input_desc =
	    inputs.isEmpty() ? nullptr : dynamic_cast<PVJsonFileDescription*>(inputs.front().get());
	if (input_desc == nullptr) {
		return {};
	}

	PVFormat format;
	if (not find_format(*input_desc, format)) {
		return {};
	}
	return format;
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
	return {"json", "jsonl", "ndjson"};
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
