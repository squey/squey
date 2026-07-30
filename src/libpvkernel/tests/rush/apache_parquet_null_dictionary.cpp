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
 * Check the import of null values held by dictionary encoded string columns.
 *
 * The parquet source asks arrow to read every STRING column as a dictionary, and arrow leaves
 * the index of a null entry unspecified while an all-null column comes with an empty dictionary.
 * Mapping those indices without checking them used to read out of the dictionary bounds, which
 * crashed on the very first row of an all-null column.
 */

#include <memory>

#include "../../plugins/common/parquet/PVParquetAPI.h"
#include "../../plugins/common/parquet/PVParquetFileDescription.h"
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <pvlogger.h>

#include <arrow/api.h>
#include <arrow/io/api.h>

#include <parquet/arrow/writer.h>

#include <optional>
#include <string>
#include <vector>

/**
 * The values below describe PATTERN_SIZE rows, repeated up to ROW_COUNT. Spanning more than the
 * 64k rows of a record batch makes sure the dictionaries are rebuilt across several chunks.
 */
static constexpr const size_t PATTERN_SIZE = 4;
static constexpr const size_t ROW_COUNT = PATTERN_SIZE * 20000; // 80000 rows, ie. 2 record batches

static constexpr const int ALL_NULL_COLUMN = 0;

/**
 * Rows holding a null value in the partially null column.
 */
static bool is_null_row(size_t row)
{
	return row % PATTERN_SIZE == 1;
}

static std::shared_ptr<arrow::Schema> test_schema()
{
	return arrow::schema({
		arrow::field("all_null", arrow::utf8()),
		arrow::field("some_null", arrow::utf8()),
		arrow::field("no_null", arrow::utf8()),
	});
}

/**
 * A null value is exported as an empty string, as convert_string does for the other columns.
 */
static const std::string null_string;

static const std::vector<std::optional<std::string>> some_null_strings = {
	"hello",
	std::nullopt,
	"héllo wörld",
	"",
};

static const std::vector<std::optional<std::string>> no_null_strings = {
	"a",
	"b",
	"c",
	"d",
};

static std::shared_ptr<arrow::Array> make_string_array(
    const std::vector<std::optional<std::string>>& pattern)
{
	arrow::StringBuilder builder;

	for (size_t row = 0; row < ROW_COUNT; row++) {
		const std::optional<std::string>& value = pattern[row % PATTERN_SIZE];
		PARQUET_THROW_NOT_OK(value ? builder.Append(*value) : builder.AppendNull());
	}

	return builder.Finish().ValueOrDie();
}

/**
 * A column whose values are all null, which arrow reads as an empty dictionary.
 */
static std::shared_ptr<arrow::Array> make_all_null_array()
{
	arrow::StringBuilder builder;

	PARQUET_THROW_NOT_OK(builder.AppendNulls(ROW_COUNT));

	return builder.Finish().ValueOrDie();
}

static void generate_parquet_file(const std::string& file_path)
{
	const std::shared_ptr<arrow::Schema>& schema = test_schema();

	std::shared_ptr<arrow::RecordBatch> record_batch = arrow::RecordBatch::Make(schema, ROW_COUNT, {
		make_all_null_array(),
		make_string_array(some_null_strings),
		make_string_array(no_null_strings),
	});

	std::shared_ptr<arrow::io::FileOutputStream> output_file =
	    arrow::io::FileOutputStream::Open(file_path).ValueOrDie();
	std::unique_ptr<parquet::arrow::FileWriter> file_writer =
	    parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), output_file).ValueOrDie();
	std::shared_ptr<arrow::Table> table = arrow::Table::FromRecordBatches({record_batch}).ValueOrDie();
	PARQUET_THROW_NOT_OK(file_writer->WriteTable(*table));
	PARQUET_THROW_NOT_OK(file_writer->Close());
}

static void import_file(
    const QString& file,
    QList<std::shared_ptr<PVRush::PVInputDescription>>& list_inputs,
    PVRush::PVFormat& format,
    PVRush::PVNraw& nraw
)
{
	PVRush::PVSourceCreator_p sc =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("parquet");
	PVRush::PVNrawOutput output(nraw);
	PVRush::PVParquetFileDescription* input_desc = new PVRush::PVParquetFileDescription(QStringList{file});
	list_inputs << PVRush::PVInputDescription_p(input_desc);
	input_desc->disable_multi_inputs(true);
	PVRush::PVParquetAPI parquet_api(input_desc);
	format = PVRush::PVFormat(parquet_api.get_format().documentElement());
	PVRush::PVExtractor extractor(format, output, sc, list_inputs);

	PVRush::PVControllerJob_p job = extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT);
	job->wait_end();
}

static void check_values(const PVRush::PVNraw& nraw)
{
	PV_VALID((size_t)nraw.row_count(), ROW_COUNT);
	PV_VALID((size_t)nraw.column_count(), test_schema()->fields().size());

	const pvcop::db::array& all_null_column = nraw.column(PVCol(ALL_NULL_COLUMN));
	const pvcop::db::array& some_null_column = nraw.column(PVCol(1));
	const pvcop::db::array& no_null_column = nraw.column(PVCol(2));

	for (size_t row = 0; row < ROW_COUNT; row++) {

		const size_t pattern_row = row % PATTERN_SIZE;

		// null values
		PV_VALID(all_null_column.is_valid(row), false);
		PV_VALID(some_null_column.is_valid(row), not is_null_row(row));
		PV_VALID(no_null_column.is_valid(row), true);

		// an all-null column yields an empty dictionary, every row being the empty string
		PV_VALID(all_null_column.at(row), null_string);

		PV_VALID(some_null_column.at(row),
		         is_null_row(row) ? null_string : *some_null_strings[pattern_row]);
		PV_VALID(no_null_column.at(row), *no_null_strings[pattern_row]);
	}
}

UNICODE_MAIN()
{
	(void) argc;
	(void) argv;

	pvtest::init_ctxt();

	const std::string& parquet_file =
	    PVRush::PVNrawCacheManager::nraw_dir().toStdString() + "/null_dictionary.parquet";
	generate_parquet_file(parquet_file);

	QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
	PVRush::PVFormat format;
	PVRush::PVNraw nraw;
	import_file(QString::fromStdString(parquet_file), list_inputs, format, nraw);

	check_values(nraw);

	// Cleanup files
	std::remove(parquet_file.c_str());

	return 0;
}
