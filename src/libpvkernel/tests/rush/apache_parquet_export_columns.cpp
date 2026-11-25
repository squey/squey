/* * MIT License
 *
 * © Squey, 2026
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

/**
 * Check that the parquet export writes the columns it is given, in their order, and the input as
 * it is when given none.
 *
 * The columns are those of the NRAW: there, the fields of a structure are columns of their own, a
 * field of a type the import leaves out is none, and a source of several files starts with the
 * column naming the file of each row.
 */

#include "../../plugins/common/parquet/PVParquetAPI.h"
#include <pvkernel/rush/PVFileListDescription.h>
#include "../../plugins/input-types/parquet/PVParquetExporter.h"
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>

#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <QDir>
#include <QTemporaryDir>

#include <memory>
#include <string>
#include <vector>

namespace
{

/**
 * Writes @p row_count rows, of ids from @p first_id on.
 */
void generate_parquet_file(const QString& path, int32_t first_id, int64_t row_count)
{
	arrow::Int32Builder ids;
	arrow::StringBuilder names;
	arrow::DoubleBuilder xs;
	arrow::DoubleBuilder ys;
	for (int64_t row = 0; row < row_count; row++) {
		const int32_t id = first_id + int32_t(row);
		PARQUET_THROW_NOT_OK(ids.Append(id));
		PARQUET_THROW_NOT_OK(names.Append("name " + std::to_string(id)));
		PARQUET_THROW_NOT_OK(xs.Append(id + 0.25));
		PARQUET_THROW_NOT_OK(ys.Append(id + 0.5));
	}

	const arrow::FieldVector position_fields = {arrow::field("x", arrow::float64()),
	                                            arrow::field("y", arrow::float64())};
	const std::shared_ptr<arrow::Schema> schema =
	    arrow::schema({arrow::field("id", arrow::int32()), arrow::field("name", arrow::utf8()),
	                   arrow::field("position", arrow::struct_(position_fields)),
	                   arrow::field("nothing", arrow::null())});
	const std::shared_ptr<arrow::Table> table = arrow::Table::Make(
	    schema, {ids.Finish().ValueOrDie(), names.Finish().ValueOrDie(),
	             arrow::StructArray::Make({xs.Finish().ValueOrDie(), ys.Finish().ValueOrDie()},
	                                      position_fields)
	                 .ValueOrDie(),
	             std::make_shared<arrow::NullArray>(row_count)});

	const std::shared_ptr<arrow::io::FileOutputStream> output_file =
	    arrow::io::FileOutputStream::Open(path.toStdString()).ValueOrDie();
	PARQUET_THROW_NOT_OK(
	    parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), output_file));
}

void import_files(const QStringList& paths,
                  bool with_input_names,
                  PVRush::PVInputType::list_inputs& inputs,
                  PVRush::PVNraw& nraw)
{
	PVRush::PVSourceCreator_p sc =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("parquet");
	PVRush::PVNrawOutput output(nraw);
	auto* input_desc = new PVRush::PVFileListDescription(paths);
	input_desc->disable_multi_inputs(not with_input_names);
	inputs << PVRush::PVInputDescription_p(input_desc);
	PVRush::PVParquetAPI parquet_api(input_desc);
	PVRush::PVFormat format(parquet_api.get_format().documentElement());
	PVRush::PVExtractor extractor(format, output, sc, inputs);

	PVRush::PVControllerJob_p job =
	    extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT);
	job->wait_end();
}

/**
 * Exports @p rows of @p nraw into @p directory, and reads the file written back.
 */
std::shared_ptr<arrow::Table> export_rows(const PVRush::PVInputType::list_inputs& inputs,
                                          const PVRush::PVNraw& nraw,
                                          const PVCore::PVColumnIndexes& columns,
                                          const std::vector<PVRow>& rows,
                                          const QDir& directory)
{
	PVCore::PVSelBitField selection(nraw.row_count());
	selection.select_none();
	for (PVRow row : rows) {
		selection.set_bit_fast(row);
	}

	const std::string path = directory.filePath("exported.parquet").toStdString();
	PVRush::PVParquetExporter exporter(inputs, nraw);
	exporter.set_columns(columns);
	exporter.export_rows(path, selection);

	const std::shared_ptr<arrow::io::ReadableFile> input =
	    arrow::io::ReadableFile::Open(path).ValueOrDie();
	std::unique_ptr<parquet::arrow::FileReader> reader =
	    parquet::arrow::OpenFile(input, arrow::default_memory_pool()).ValueOrDie();
	return reader->ReadTable().ValueOrDie()->CombineChunks().ValueOrDie();
}

std::string names_of(const std::shared_ptr<arrow::Table>& table)
{
	std::string names;
	for (const std::string& name : table->schema()->field_names()) {
		names += (names.empty() ? "" : ",") + name;
	}
	return names;
}

template <class Array>
std::shared_ptr<Array> column_of(const std::shared_ptr<arrow::Table>& table, int column)
{
	return std::static_pointer_cast<Array>(table->column(column)->chunk(0));
}

std::string text_of(const std::shared_ptr<arrow::Table>& table, int column, int64_t row)
{
	// The import reads text as dictionaries, which the export writes as they are.
	const std::shared_ptr<arrow::Array> text =
	    arrow::compute::Cast(*table->column(column)->chunk(0), arrow::utf8()).ValueOrDie();
	return std::static_pointer_cast<arrow::StringArray>(text)->GetString(row);
}

} // namespace

UNICODE_MAIN()
{
	(void)argc;
	(void)argv;

	pvtest::init_ctxt();

	QTemporaryDir temporary_dir(PVRush::PVNrawCacheManager::nraw_dir() + "/export_columns_XXXXXX");
	PV_ASSERT_VALID(temporary_dir.isValid());
	const QDir root(temporary_dir.path());
	generate_parquet_file(root.filePath("first.parquet"), 0, 5);
	generate_parquet_file(root.filePath("second.parquet"), 10, 3);

	{
		PVRush::PVInputType::list_inputs inputs;
		PVRush::PVNraw nraw;
		import_files({root.filePath("first.parquet")}, false, inputs, nraw);
		// id, name, position.x and position.y
		PV_VALID((size_t)nraw.column_count(), size_t(4));

		const std::shared_ptr<arrow::Table> chosen =
		    export_rows(inputs, nraw, {PVCol(3), PVCol(1), PVCol(0), PVCol(1)}, {1, 3}, root);
		PV_VALID(names_of(chosen), std::string("position.y,name,id"));
		PV_VALID(chosen->schema()->field(0)->type()->ToString(), std::string("double"));
		PV_VALID(chosen->num_rows(), int64_t(2));
		PV_VALID(column_of<arrow::DoubleArray>(chosen, 0)->Value(1), 3.5);
		PV_VALID(text_of(chosen, 1, 0), std::string("name 1"));
		PV_VALID(column_of<arrow::Int32Array>(chosen, 2)->Value(1), int32_t(3));

		const std::shared_ptr<arrow::Table> whole = export_rows(inputs, nraw, {}, {1, 3}, root);
		PV_VALID(names_of(whole), std::string("id,name,position,nothing"));
		PV_VALID(whole->num_rows(), int64_t(2));
		PV_VALID(column_of<arrow::Int32Array>(whole, 0)->Value(0), int32_t(1));
	}

	{
		PVRush::PVInputType::list_inputs inputs;
		PVRush::PVNraw nraw;
		import_files({root.filePath("first.parquet"), root.filePath("second.parquet")}, true,
		             inputs, nraw);
		// filename, id, name, position.x and position.y
		PV_VALID((size_t)nraw.column_count(), size_t(5));

		const std::shared_ptr<arrow::Table> named =
		    export_rows(inputs, nraw, {PVCol(0), PVCol(1)}, {0, 4, 5, 7}, root);
		PV_VALID(names_of(named), std::string("filename,id"));
		const std::shared_ptr<arrow::Int32Array> ids = column_of<arrow::Int32Array>(named, 1);
		const std::vector<std::pair<std::string, int32_t>> expected = {{"first.parquet", 0},
		                                                               {"first.parquet", 4},
		                                                               {"second.parquet", 10},
		                                                               {"second.parquet", 12}};
		PV_VALID(named->num_rows(), int64_t(expected.size()));
		for (size_t row = 0; row < expected.size(); row++) {
			PV_VALID(text_of(named, 0, int64_t(row)), expected[row].first, "row", row);
			PV_VALID(ids->Value(int64_t(row)), expected[row].second, "row", row);
		}
	}

	return 0;
}
