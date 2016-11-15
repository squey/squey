/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVBASE_TYPES_H
#define PVBASE_TYPES_H

#include <limits>

#include "typed_builtin.h"
#include <pvlogger.h>

class PVCol : public __impl::PVTypedBuiltin<PVCol>
{
  public:
	using __impl::PVTypedBuiltin<PVCol>::PVTypedBuiltin;
};

namespace std
{
template <>
struct hash<PVCol> {
	size_t operator()(const PVCol& c) const { return std::hash<typename PVCol::value_type>()(c); }
};
}

class PVCombCol : public __impl::PVTypedBuiltin<PVCombCol>
{
  public:
	using __impl::PVTypedBuiltin<PVCombCol>::PVTypedBuiltin;
};

namespace std
{
template <>
struct hash<PVCombCol> {
	size_t operator()(const PVCombCol& c) const
	{
		return std::hash<typename PVCombCol::value_type>()(c);
	}
};
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
