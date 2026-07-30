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
 * Check which columns the parquet source asks arrow to read as dictionaries.
 *
 * set_read_dictionary() indexes the parquet leaf columns, and a map contributes two of them
 * (its key and its item) while the flattened arrow schema only counts it as a single field.
 * Deriving the index from the flattened schema therefore shifts it as soon as a map is met,
 * and ends up dictionary encoding an unrelated leaf.
 *
 * The schema below makes that shift observable. Its flattened fields are m1(0), m2(1) and
 * text(2), whereas its parquet leaves are m1.key(0), m1.value(1), m2.key(2), m2.value(3) and
 * text(4) : the index of the only string field, 2, designates the key of the second map.
 * Once that key is read as a dictionary, convert_complex_type_as_string() renders it through
 * a DictionaryScalar, whose ToString() dumps the whole dictionary over several lines. The
 * second map then holds "{[" followed by that dump, embedded end of lines included.
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

#include <string>

static constexpr const size_t ROW_COUNT = 1000;

static std::shared_ptr<arrow::Schema> test_schema()
{
	return arrow::schema({
		arrow::field("m1",   arrow::map(arrow::utf8(), arrow::utf8())),
		arrow::field("m2",   arrow::map(arrow::utf8(), arrow::utf8())),
		arrow::field("text", arrow::utf8()),
	});
}

/**
 * A single entry per row, whose key and value are the same on every row : the column is
 * dictionary encoded by parquet either way, only the way it is read back differs.
 */
static std::shared_ptr<arrow::Array> make_map_array(const std::string& key, const std::string& value)
{
	auto key_builder = std::make_shared<arrow::StringBuilder>();
	auto item_builder = std::make_shared<arrow::StringBuilder>();
	arrow::MapBuilder builder(arrow::default_memory_pool(), key_builder, item_builder);

	for (size_t row = 0; row < ROW_COUNT; row++) {
		PARQUET_THROW_NOT_OK(builder.Append());
		PARQUET_THROW_NOT_OK(key_builder->Append(key));
		PARQUET_THROW_NOT_OK(item_builder->Append(value));
	}

	return builder.Finish().ValueOrDie();
}

static std::shared_ptr<arrow::Array> make_string_array(const std::string& value)
{
	arrow::StringBuilder builder;

	for (size_t row = 0; row < ROW_COUNT; row++) {
		PARQUET_THROW_NOT_OK(builder.Append(value));
	}

	return builder.Finish().ValueOrDie();
}

static void generate_parquet_file(const std::string& file_path)
{
	const std::shared_ptr<arrow::Schema>& schema = test_schema();

	std::shared_ptr<arrow::RecordBatch> record_batch = arrow::RecordBatch::Make(schema, ROW_COUNT, {
		make_map_array("k1", "v1"),
		make_map_array("k2", "v2"),
		make_string_array("text"),
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

	const pvcop::db::array& m1_column = nraw.column(PVCol(0));
	const pvcop::db::array& m2_column = nraw.column(PVCol(1));
	const pvcop::db::array& text_column = nraw.column(PVCol(2));

	for (size_t row = 0; row < ROW_COUNT; row++) {
		// Both maps must be rendered the same way, which only holds as long as no map leaf
		// is turned into a dictionary behind their back.
		PV_VALID(m1_column.at(row), std::string("{\"k1\":v1}"));
		PV_VALID(m2_column.at(row), std::string("{\"k2\":v2}"));
		PV_VALID(text_column.at(row), std::string("text"));
	}
}

UNICODE_MAIN()
{
	(void) argc;
	(void) argv;

	pvtest::init_ctxt();

	const std::string& parquet_file =
	    PVRush::PVNrawCacheManager::nraw_dir().toStdString() + "/dictionary_column_index.parquet";
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
