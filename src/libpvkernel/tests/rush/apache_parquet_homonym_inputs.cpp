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
 * Check the column that names the file of each row, when a source concatenates several parquet
 * files.
 *
 * The rows refer to the names by their index in a dictionary, which keeps a single entry per
 * distinct name. Files of the same name, from different directories or listed twice, must neither
 * show the name of another file nor refer past the end of the dictionary.
 */

#include <memory>

#include "../../plugins/common/parquet/PVParquetAPI.h"
#include "../../plugins/common/parquet/PVParquetFileDescription.h"
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <arrow/api.h>
#include <arrow/io/api.h>

#include <parquet/arrow/writer.h>

#include <QDir>
#include <QTemporaryDir>

#include <iostream>
#include <map>
#include <string>

/**
 * More rows than the 64k of a record batch, so that a file spans several chunks.
 */
static constexpr const size_t ROW_COUNT = 70000;

static constexpr const int NAME_COLUMN = 0;
static constexpr const int ID_COLUMN = 1;

/**
 * The files written for the test, relative to its directory, with the value of all their rows.
 */
static const std::map<QString, int64_t> file_ids = {
    {"a/same.parquet", 0},
    {"b/same.parquet", 1},
    {"b/other.parquet", 2},
};

static void generate_parquet_file(const QString& file_path, int64_t id)
{
	const std::shared_ptr<arrow::Schema> schema =
	    arrow::schema({arrow::field("id", arrow::int64())});

	arrow::Int64Builder builder;
	for (size_t row = 0; row < ROW_COUNT; row++) {
		PARQUET_THROW_NOT_OK(builder.Append(id));
	}
	std::shared_ptr<arrow::RecordBatch> record_batch =
	    arrow::RecordBatch::Make(schema, ROW_COUNT, {builder.Finish().ValueOrDie()});

	std::shared_ptr<arrow::io::FileOutputStream> output_file =
	    arrow::io::FileOutputStream::Open(file_path.toStdString()).ValueOrDie();
	std::unique_ptr<parquet::arrow::FileWriter> file_writer =
	    parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), output_file)
	        .ValueOrDie();
	std::shared_ptr<arrow::Table> table =
	    arrow::Table::FromRecordBatches({record_batch}).ValueOrDie();
	PARQUET_THROW_NOT_OK(file_writer->WriteTable(*table));
	PARQUET_THROW_NOT_OK(file_writer->Close());
}

static void import_files(const QStringList& paths, PVRush::PVNraw& nraw)
{
	PVRush::PVSourceCreator_p sc =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("parquet");
	PVRush::PVNrawOutput output(nraw);
	auto* input_desc = new PVRush::PVParquetFileDescription(paths);
	QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
	list_inputs << PVRush::PVInputDescription_p(input_desc);
	PVRush::PVParquetAPI parquet_api(input_desc);
	PVRush::PVFormat format(parquet_api.get_format().documentElement());
	PVRush::PVExtractor extractor(format, output, sc, list_inputs);

	PVRush::PVControllerJob_p job =
	    extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT);
	job->wait_end();
}

/**
 * Imports @p files, relative to @p root, as a single source, and checks that the rows of each of
 * them are named as @p names says.
 */
static void check_names(const QDir& root, const QStringList& files, const QStringList& names)
{
	std::cout << "files: " << files.join(", ").toStdString() << std::endl;

	QStringList paths;
	for (const QString& file : files) {
		paths << root.filePath(file);
	}
	PVRush::PVNraw nraw;
	import_files(paths, nraw);

	PV_VALID((size_t)nraw.row_count(), (size_t)files.size() * ROW_COUNT);
	PV_VALID((size_t)nraw.column_count(), (size_t)2);

	const pvcop::db::array& name_column = nraw.column(PVCol(NAME_COLUMN));
	const pvcop::db::array& id_column = nraw.column(PVCol(ID_COLUMN));
	for (size_t row = 0; row < nraw.row_count(); row++) {
		const auto position = static_cast<qsizetype>(row / ROW_COUNT);
		PV_VALID(id_column.at(row), std::to_string(file_ids.at(files[position])), "row", row);
		PV_VALID(name_column.at(row), names[position].toStdString(), "row", row);
	}
}

UNICODE_MAIN()
{
	(void)argc;
	(void)argv;

	pvtest::init_ctxt();

	QTemporaryDir temporary_dir(PVRush::PVNrawCacheManager::nraw_dir() + "/homonym_inputs_XXXXXX");
	PV_ASSERT_VALID(temporary_dir.isValid());
	const QDir root(temporary_dir.path());
	PV_ASSERT_VALID(root.mkpath("a") and root.mkpath("b"));
	for (const auto& [file, id] : file_ids) {
		generate_parquet_file(root.filePath(file), id);
	}

	// Different file names, even from different directories, name their files.
	check_names(root, {"a/same.parquet", "b/other.parquet"}, {"same.parquet", "other.parquet"});

	// A file listed twice takes a single entry of the dictionary.
	check_names(root, {"b/same.parquet", "b/same.parquet", "b/other.parquet"},
	            {"same.parquet", "same.parquet", "other.parquet"});

	// Files of the same name are told apart by their paths from the directory holding them all.
	check_names(root, {"a/same.parquet", "b/same.parquet", "b/other.parquet"},
	            {"a/same.parquet", "b/same.parquet", "b/other.parquet"});

	return 0;
}
