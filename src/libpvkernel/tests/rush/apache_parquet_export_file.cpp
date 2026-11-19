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
 * Check that each compression codec the parquet export offers writes a file that reads back:
 * Arrow knows codecs that Parquet refuses to write.
 */

#include "../../plugins/common/parquet/PVParquetFileDescription.h"
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
#include <QTemporaryDir>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

static constexpr const int64_t ROW_COUNT = 1000;

UNICODE_MAIN()
{
	(void)argc;
	(void)argv;

	pvtest::init_ctxt();

	QTemporaryDir temporary_dir(PVRush::PVNrawCacheManager::nraw_dir() + "/export_file_XXXXXX");
	PV_ASSERT_VALID(temporary_dir.isValid());
	const QDir root(temporary_dir.path());

	arrow::Int64Builder ids;
	for (int64_t row = 0; row < ROW_COUNT; row++) {
		PARQUET_THROW_NOT_OK(ids.Append(row));
	}
	const std::shared_ptr<arrow::Table> input = arrow::Table::Make(
	    arrow::schema({arrow::field("id", arrow::int64())}), {ids.Finish().ValueOrDie()});
	const QString input_path = root.filePath("input.parquet");
	PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(
	    *input, arrow::default_memory_pool(),
	    arrow::io::FileOutputStream::Open(input_path.toStdString()).ValueOrDie()));

	PVRush::PVInputType::list_inputs inputs;
	inputs << PVRush::PVInputDescription_p(new PVRush::PVParquetFileDescription({input_path}));
	// The export of the input as it is reads its rows from the input, and none from the NRAW.
	const PVRush::PVNraw nraw;
	PVCore::PVSelBitField selection(ROW_COUNT);
	selection.select_all();

	const std::vector<arrow::Compression::type> codecs =
	    PVRush::PVParquetExporter::compression_codecs();
	PV_ASSERT_VALID(std::find(codecs.begin(), codecs.end(), arrow::Compression::UNCOMPRESSED) !=
	                codecs.end());

	for (arrow::Compression::type codec : codecs) {
		const std::string name = arrow::util::Codec::GetCodecAsString(codec);
		const std::string path =
		    root.filePath(QString::fromStdString(name) + ".parquet").toStdString();
		PVRush::PVParquetExporter exporter(inputs, nraw);
		exporter.set_compression_codec(codec);
		exporter.export_rows(path, selection);

		std::unique_ptr<parquet::arrow::FileReader> reader =
		    parquet::arrow::OpenFile(arrow::io::ReadableFile::Open(path).ValueOrDie(),
		                             arrow::default_memory_pool())
		        .ValueOrDie();
		PV_VALID(reader->parquet_reader()->metadata()->RowGroup(0)->ColumnChunk(0)->compression(),
		         codec, "codec", name);
		PV_ASSERT_VALID(reader->ReadTable().ValueOrDie()->Equals(*input), "codec", name);
	}

	return 0;
}
