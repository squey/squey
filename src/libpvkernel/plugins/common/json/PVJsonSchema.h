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

#ifndef __RUSH_PVJSONSCHEMA_H__
#define __RUSH_PVJSONSCHEMA_H__

#include <arrow/type_fwd.h>

#include <QString>
#include <QStringList>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace PVRush
{

/**
 * The columns that the import makes of the values of the JSON objects of files, one object per
 * line. The fields of the objects of a field are columns of their own, named after the fields
 * leading to them, joined by dots.
 *
 * A column of values of a single type has that type, but for integers among other numbers, which
 * are numbers, and for strings that are all times in ISO 8601, which are times. The values of the
 * other columns are their JSON text, but for strings: arrays, and values of mixed types. Fields
 * that are always null have no column, nor objects without fields.
 */
class PVJsonSchema
{
  public:
	enum class Type { BOOLEAN, INTEGER, NUMBER, TIME, TEXT };

	struct Column {
		/** The fields leading to the value, from the object of a line. */
		std::vector<std::string> fields;
		/** The fields joined by dots. */
		std::string name;
		Type type;
		/** The count of the objects that give it a value, null aside. */
		size_t value_count;
	};

	/** Gets the bytes read of the files and their size, and stops the reading on false. */
	using progress_f = std::function<bool(size_t bytes_read, size_t size)>;

  public:
	/**
	 * The schema of the objects of the files @p paths, or nullptr if @p progress stopped reading
	 * them.
	 *
	 * @throw std::runtime_error if a file cannot be read
	 */
	static std::shared_ptr<const PVJsonSchema> infer(const QStringList& paths,
	                                                 const progress_f& progress = {});

	/**
	 * The microseconds since the epoch of @p time, written in ISO 8601, of which it can have up
	 * to nanoseconds. It is in UTC, unless it gives a time zone.
	 */
	static std::optional<int64_t> microseconds_of(std::string_view time);

  public:
	const std::vector<Column>& columns() const { return _columns; }

	/**
	 * A field of the type of each column, of its name, after the dictionary of the names of the
	 * files that a source of several of them starts with, if @p input_names.
	 */
	std::shared_ptr<arrow::Schema> arrow_schema(bool input_names = false) const;

	/** This schema, with only the columns named @p names among its own, or all of them if none. */
	std::shared_ptr<const PVJsonSchema> with_columns(const QStringList& names) const;

	/** The count of the objects of the files. */
	size_t row_count() const { return _row_count; }

	/** The count of the lines of the files that are not JSON objects, which have no row. */
	size_t invalid_line_count() const { return _invalid_line_count; }

	/** The file of the first line that is not a JSON object. */
	const QString& first_invalid_file() const { return _first_invalid_file; }

	/** The number, from 1, of the first line that is not a JSON object. */
	size_t first_invalid_line() const { return _first_invalid_line; }

  private:
	std::vector<Column> _columns;
	size_t _row_count = 0;
	size_t _invalid_line_count = 0;
	QString _first_invalid_file;
	size_t _first_invalid_line = 0;
};

} // namespace PVRush

#endif // __RUSH_PVJSONSCHEMA_H__
