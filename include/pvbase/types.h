/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
		type() : type(std::numeric_limits<underlying_type>::max()){};                              \
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
