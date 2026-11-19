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

#ifndef __RUSH_PVJSONROWS_H__
#define __RUSH_PVJSONROWS_H__

#include "PVJsonSchema.h"

#include <arrow/array/builder_base.h>
#include <arrow/type_fwd.h>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace PVRush
{

/**
 * The rows that the JSON objects of lines make of the columns of a schema, as Arrow arrays.
 *
 * A field out of the schema is left out, as is the second value of a field in an object. A value
 * that its column cannot hold, as a file changed since its schema was found, is a null.
 */
class PVJsonRows
{
  public:
	/** @p schema has to outlive the rows. */
	explicit PVJsonRows(const PVJsonSchema& schema);
	~PVJsonRows();

  public:
	/**
	 * Appends the row of the JSON object of @p line, a null-terminated line, which the parser
	 * writes to as it reads it.
	 *
	 * @return whether @p line is a JSON object, as nothing is appended otherwise
	 */
	bool append(char* line);

	size_t row_count() const { return _row_count; }

	/**
	 * The arrays of the rows appended since the previous call, one for each column of the schema,
	 * in its order.
	 */
	arrow::ArrayVector finish();

  private:
	struct Node;

	void add_fields(const Node& node, const rapidjson::Value& object);
	void add_value(size_t column, const rapidjson::Value& value);

  private:
	const PVJsonSchema& _schema;
	std::unique_ptr<Node> _root;
	std::vector<std::unique_ptr<arrow::ArrayBuilder>> _builders;
	rapidjson::Document _document;
	rapidjson::StringBuffer _text;
	size_t _row_count = 0;
};

} // namespace PVRush

#endif // __RUSH_PVJSONROWS_H__
