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
 * Check the file the parquet export writes:
 *
 * * each compression codec the export offers writes a file that reads back, as Arrow knows codecs
 *   that Parquet refuses to write;
 * * a canceled export reads no further input, and leaves no file behind;
 * * a failure, to write the file or to read an input, reports itself as an export error, which
 *   the export dialogs show.
 */

#include <pvkernel/rush/PVFileListDescription.h>
#include "../../plugins/input-types/parquet/PVParquetExporter.h"
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/util/compression.h>

#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace
{

constexpr const int64_t ROW_COUNT = 1000;

/**
 * Writes the ids from @p first_id on to @p path, and returns them.
 */
std::shared_ptr<arrow::Table> write_ids(const QString& path, int64_t first_id)
{
	arrow::Int64Builder ids;
	for (int64_t row = 0; row < ROW_COUNT; row++) {
		PARQUET_THROW_NOT_OK(ids.Append(first_id + row));
	}
	const std::shared_ptr<arrow::Table> table = arrow::Table::Make(
	    arrow::schema({arrow::field("id", arrow::int64())}), {ids.Finish().ValueOrDie()});
	PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(
	    *table, arrow::default_memory_pool(),
	    arrow::io::FileOutputStream::Open(path.toStdString()).ValueOrDie()));
	return table;
}

PVRush::PVInputType::list_inputs inputs_of(const QStringList& paths)
{
	PVRush::PVInputType::list_inputs inputs;
	inputs << PVRush::PVInputDescription_p(new PVRush::PVFileListDescription(paths));
	return inputs;
}

PVCore::PVSelBitField all_rows(size_t row_count)
{
	PVCore::PVSelBitField selection(row_count);
	selection.select_all();
	return selection;
}

/**
 * Whether the export of the @p row_count rows of @p inputs to @p path, with @p codec, fails.
 */
bool fails(const PVRush::PVInputType::list_inputs& inputs,
           size_t row_count,
           const QString& path,
           arrow::Compression::type codec = arrow::Compression::UNCOMPRESSED)
{
	const PVRush::PVNraw nraw;
	PVRush::PVParquetExporter exporter(inputs, nraw);
	exporter.set_compression_codec(codec);
	try {
		exporter.export_rows(path.toStdString(), all_rows(row_count));
	} catch (const PVRush::PVExportError& e) {
		std::cout << "export error: " << e.what() << std::endl;
		return true;
	}
	return false;
}

} // namespace

UNICODE_MAIN()
{
	(void)argc;
	(void)argv;

	pvtest::init_ctxt();

	QTemporaryDir temporary_dir(PVRush::PVNrawCacheManager::nraw_dir() + "/export_file_XXXXXX");
	PV_ASSERT_VALID(temporary_dir.isValid());
	const QDir root(temporary_dir.path());

	const QString first_path = root.filePath("first.parquet");
	const QString second_path = root.filePath("second.parquet");
	const std::shared_ptr<arrow::Table> first = write_ids(first_path, 0);
	write_ids(second_path, ROW_COUNT);

	// The export of the inputs as they are reads their rows from them, and none from the NRAW.
	const PVRush::PVNraw nraw;

	// The exporters keep a reference to their inputs.
	const PVRush::PVInputType::list_inputs first_inputs = inputs_of({first_path});
	const PVRush::PVInputType::list_inputs both_inputs = inputs_of({first_path, second_path});

	const std::vector<arrow::Compression::type> codecs =
	    PVRush::PVParquetExporter::compression_codecs();
	PV_ASSERT_VALID(std::find(codecs.begin(), codecs.end(), arrow::Compression::UNCOMPRESSED) !=
	                codecs.end());
	for (arrow::Compression::type codec : codecs) {
		const std::string name = arrow::util::Codec::GetCodecAsString(codec);
		const std::string path =
		    root.filePath(QString::fromStdString(name) + ".parquet").toStdString();
		PVRush::PVParquetExporter exporter(first_inputs, nraw);
		exporter.set_compression_codec(codec);
		exporter.export_rows(path, all_rows(ROW_COUNT));

		std::unique_ptr<parquet::arrow::FileReader> reader =
		    parquet::arrow::OpenFile(arrow::io::ReadableFile::Open(path).ValueOrDie(),
		                             arrow::default_memory_pool())
		        .ValueOrDie();
		PV_VALID(reader->parquet_reader()->metadata()->RowGroup(0)->ColumnChunk(0)->compression(),
		         codec, "codec", name);
		PV_ASSERT_VALID(reader->ReadTable().ValueOrDie()->Equals(*first), "codec", name);
	}

	{
		PVRush::PVParquetExporter exporter(both_inputs, nraw);
		size_t progress_count = 0;
		exporter.set_progress_func([&](size_t) {
			progress_count++;
			exporter.cancel();
		});
		const QString path = root.filePath("canceled.parquet");
		exporter.export_rows(path.toStdString(), all_rows(2 * ROW_COUNT));
		// Each file is a single batch of rows.
		PV_VALID(progress_count, size_t(1));
		PV_ASSERT_VALID(not QFileInfo::exists(path));
	}

	PV_ASSERT_VALID(fails(first_inputs, ROW_COUNT, first_path));
	PV_ASSERT_VALID(fails(first_inputs, ROW_COUNT, root.filePath("missing/exported.parquet")));
	if (arrow::util::Codec::IsAvailable(arrow::Compression::BZ2)) {
		PV_ASSERT_VALID(
		    fails(first_inputs, ROW_COUNT, root.filePath("bz2.parquet"), arrow::Compression::BZ2));
	}

	// Inputs removed since the import, at the start and after the first file.
	const QString removed_path = root.filePath("removed.parquet");
	write_ids(removed_path, 0);
	const PVRush::PVInputType::list_inputs removed_inputs = inputs_of({removed_path});
	const PVRush::PVInputType::list_inputs second_removed_inputs =
	    inputs_of({first_path, removed_path});
	QFile::remove(removed_path);
	PV_ASSERT_VALID(fails(removed_inputs, ROW_COUNT, root.filePath("first_removed.parquet")));
	PV_ASSERT_VALID(
	    fails(second_removed_inputs, 2 * ROW_COUNT, root.filePath("second_removed.parquet")));

	return 0;
}
