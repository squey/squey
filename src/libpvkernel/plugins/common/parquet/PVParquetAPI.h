/* * MIT License
 *
 * © Squey, 2024
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

#ifndef __RUSH_PVPARQUETAPI_H__
#define __RUSH_PVPARQUETAPI_H__

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <pvkernel/core/serialize_numbers.h>
#include <parquet/arrow/reader.h>
#include <arrow/type.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>
#include <qlist.h>
#include <qstring.h>
#include <stddef.h>
#include <QDomDocument>
#include <QTextStream>
#include <unordered_map>
#include <functional>
#include <memory>
#include <vector>

#include "PVParquetFileDescription.h"

namespace PVRush
{

/******************************************************************************
 *
 * PVRush::PVParquetAPI
 *
 *****************************************************************************/


class PVParquetAPI
{
  public:
	struct pvcop_type_infos {
		size_t size_in_bytes;
		const char* string;
	};
	static const std::unordered_map<arrow::Type::type, PVParquetAPI::pvcop_type_infos>  pvcop_types_map;

  public:
	PVParquetAPI(const PVRush::PVParquetFileDescription* input_desc);

	parquet::arrow::FileReader* arrow_reader() { return _arrow_reader.get(); }
	const parquet::arrow::FileReader* arrow_reader() const { return _arrow_reader.get(); }

	size_t files_count() const { return _input_desc->paths().size(); }
	bool next_file();
	void reset() { _next_input_file = 0; }
	void visit_files(const std::function<void()>& f);

	size_t row_count() const { return _arrow_reader->parquet_reader()->metadata()->num_rows(); }
	const std::vector<size_t>& column_indexes() const { return _column_indexes; }
	bool multi_inputs() const { return files_count() > 1 and _input_desc->multi_inputs(); }
	bool is_bit_optimizable() const;
	bool same_schemas() const;
	static std::shared_ptr<arrow::Schema> flatten_schema(const std::shared_ptr<arrow::Schema>& schema);
	static std::shared_ptr<arrow::Table> flatten_table(const std::shared_ptr<arrow::Table>& table);

  public:
	QDomDocument get_format();

  private:
  	std::unique_ptr<parquet::arrow::FileReader> _arrow_reader;
	mutable QDomDocument _format;
	const PVRush::PVParquetFileDescription* _input_desc = nullptr;
	size_t _next_input_file = 0;
	std::vector<size_t> _column_indexes;
};

} // namespace PVRush

#endif // __RUSH_PVPARQUETAPI_H__
