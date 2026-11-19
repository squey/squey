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

#ifndef __PVRUSH_PVPARQUETEXPORTERBASE_H__
#define __PVRUSH_PVPARQUETEXPORTERBASE_H__

#include <pvkernel/core/PVColumnIndexes.h>
#include <pvkernel/rush/PVExporter.h>

#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type_fwd.h>
#include <arrow/util/type_fwd.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace parquet::arrow
{
class FileWriter;
} // namespace parquet::arrow

namespace PVRush
{

/**
 * What the Parquet exporters share: the codec of their file, the choice of its columns, and the
 * writing of the file.
 */
class PVParquetExporterBase : public PVExporterBase
{
  public:
	/** The codecs of Arrow that Parquet supports, which are the only ones writing a file. */
	static std::vector<arrow::Compression::type> compression_codecs();

	/**
	 * @throw PVExportError if @p status reports a failure, which the export dialogs show
	 */
	static void check(const arrow::Status& status);

	/**
	 * @throw PVExportError if @p result is a failure
	 */
	template <typename T>
	static T value_of(arrow::Result<T> result)
	{
		check(result.status());
		return std::move(result).ValueUnsafe();
	}

  public:
	bool exports_columns() const override { return true; }

	/**
	 * Keeps each column at its first place only, as readers find the columns of a Parquet file
	 * by their names.
	 */
	void set_columns(const PVCore::PVColumnIndexes& columns) override;

	void set_compression_codec(arrow::Compression::type codec) { _compression_codec = codec; }
	arrow::Compression::type compression_codec() const { return _compression_codec; }

  protected:
	/**
	 * Creates the file @p path, for rows of @p schema. The file records @p schema as well, which
	 * gives readers back the Arrow types that Parquet does not have.
	 *
	 * @throw PVExportError if the file cannot be created
	 */
	std::unique_ptr<parquet::arrow::FileWriter>
	open(const std::string& path, const std::shared_ptr<arrow::Schema>& schema) const;

	/**
	 * If the export was canceled, closes @p writer, and removes the file @p path it was writing.
	 *
	 * @return whether the export was canceled
	 */
	bool close_if_canceled(parquet::arrow::FileWriter& writer, const std::string& path);

  protected:
	/** Empty for the input as it is. */
	PVCore::PVColumnIndexes _columns;

  private:
	arrow::Compression::type _compression_codec = arrow::Compression::UNCOMPRESSED;
};

} // namespace PVRush

#endif // __PVRUSH_PVPARQUETEXPORTERBASE_H__
