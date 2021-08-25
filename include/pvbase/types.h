/* * MIT License
 *
 * © ESI Group, 2015
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

#ifndef PVBASE_TYPES_H
#define PVBASE_TYPES_H

#include <limits>

#include <QtGlobal>
#include <QMetaType>

#include <type_safe/strong_typedef.hpp>

#include <pvlogger.h>

#define DEFINE_STRONG_TYPEDEF(type, underlying_type)                                               \
                                                                                                   \
	struct type : type_safe::strong_typedef<type, underlying_type>,                                \
	              type_safe::strong_typedef_op::integer_arithmetic<type> {                         \
		using type_safe::strong_typedef<type, underlying_type>::strong_typedef;                    \
		using value_type = underlying_type;                                                        \
                                                                                                   \
		constexpr type() : type(std::numeric_limits<underlying_type>::max()){};                    \
                                                                                                   \
		constexpr operator const value_type&() const noexcept                                      \
		{                                                                                          \
			return type_safe::strong_typedef<type, underlying_type>::operator const value_type&(); \
		}                                                                                          \
                                                                                                   \
		const value_type& value() const noexcept { return operator const value_type&(); }          \
	};                                                                                             \
                                                                                                   \
	namespace std                                                                                  \
	{                                                                                              \
	template <>                                                                                    \
	struct hash<type> {                                                                            \
		size_t operator()(const type& c) const                                                     \
		{                                                                                          \
			return std::hash<type::value_type>()(c.value());                                       \
		}                                                                                          \
	};                                                                                             \
	}

DEFINE_STRONG_TYPEDEF(PVCol, int)
DEFINE_STRONG_TYPEDEF(PVCombCol, int)

Q_DECLARE_METATYPE(PVCol)
Q_DECLARE_METATYPE(PVCombCol)

template <class Stream>
Stream& operator<<(Stream& stream, PVCol col)
{
	return stream << static_cast<PVCol::value_type>(col);
}

template <class Stream>
Stream& operator>>(Stream& stream, PVCol col)
{
	return stream >> static_cast<PVCol::value_type&>(col);
}

using PVRow = quint32;
using chunk_index = quint64;

static constexpr const PVRow PVROW_INVALID_VALUE = std::numeric_limits<PVRow>::max();

// Maximum row count that can be read from inputs by the import pipeline
static constexpr const uint64_t IMPORT_PIPELINE_ROW_COUNT_LIMIT =
    std::numeric_limits<uint64_t>::max();

// Maximum row count that can be loaded by the application (ie. neither invalid nor filtered)
static constexpr const uint64_t EXTRACTED_ROW_COUNT_LIMIT = std::numeric_limits<int32_t>::max();

// Define necessary alignement of pointers of PVRows for vectorisation usage
static constexpr const PVRow PVROW_VECTOR_ALIGNEMENT = (128 / (sizeof(PVRow) * 8));

#define DECLARE_ALIGN(n) __attribute__((aligned(n)))

#endif /* PVBASE_TYPES_H */
