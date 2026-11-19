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

#include "PVParquetExporterBase.h"

#include <arrow/filesystem/filesystem.h>
#include <arrow/io/interfaces.h>
#include <arrow/memory_pool.h>
#include <arrow/util/compression.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>
#include <parquet/types.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>

std::vector<arrow::Compression::type> PVRush::PVParquetExporterBase::compression_codecs()
{
	std::vector<arrow::Compression::type> codecs;
	for (int i = 0; i < std::numeric_limits<uint8_t>::max(); i++) {
		const auto codec = arrow::Compression::type(i);
		if (arrow::util::Codec::IsAvailable(codec) and parquet::IsCodecSupported(codec)) {
			codecs.push_back(codec);
		}
	}
	return codecs;
}

void PVRush::PVParquetExporterBase::check(const arrow::Status& status)
{
	if (not status.ok()) {
		throw PVExportError(status.ToString());
	}
}

void PVRush::PVParquetExporterBase::set_columns(const PVCore::PVColumnIndexes& columns)
{
	_columns.clear();
	for (PVCol column : columns) {
		if (std::find(_columns.begin(), _columns.end(), column) == _columns.end()) {
			_columns.push_back(column);
		}
	}
}

std::unique_ptr<parquet::arrow::FileWriter>
PVRush::PVParquetExporterBase::open(const std::string& path,
                                    const std::shared_ptr<arrow::Schema>& schema) const
{
	const std::shared_ptr<parquet::WriterProperties> parquet_props =
	    parquet::WriterProperties::Builder().compression(_compression_codec)->build();
	const std::shared_ptr<parquet::ArrowWriterProperties> arrow_props =
	    parquet::ArrowWriterProperties::Builder().store_schema()->build();
	const std::shared_ptr<arrow::fs::FileSystem> fs =
	    value_of(arrow::fs::FileSystemFromUri("file:///"));
	return value_of(parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(),
	                                                 value_of(fs->OpenOutputStream(path)),
	                                                 parquet_props, arrow_props));
}

bool PVRush::PVParquetExporterBase::close_if_canceled(parquet::arrow::FileWriter& writer,
                                                      const std::string& path)
{
	if (not _canceled) {
		return false;
	}
	check(writer.Close());
	std::remove(path.c_str());
	_canceled = false;
	return true;
}
