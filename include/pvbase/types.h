/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVBASE_TYPES_H
#define PVBASE_TYPES_H

#include <QtGlobal>

using PVCol = qint32;
using PVRow = quint32;
using chunk_index = quint64;

static constexpr const PVRow PVROW_INVALID_VALUE = std::numeric_limits<PVRow>::max();
static constexpr const PVCol PVCOL_INVALID_VALUE = std::numeric_limits<PVCol>::max();

// Maximum row count that can be read from inputs by the import pipeline
static constexpr const uint64_t IMPORT_PIPELINE_ROW_COUNT_LIMIT =
    std::numeric_limits<uint64_t>::max();

// Maximum row count that can be loaded by the application (ie. neither invalid nor filtered)
static constexpr const uint64_t EXTRACTED_ROW_COUNT_LIMIT = std::numeric_limits<int32_t>::max();

// Define necessary alignement of pointers of PVRows for vectorisation usage
static constexpr const PVRow PVROW_VECTOR_ALIGNEMENT = (128 / (sizeof(PVRow) * 8));

#define DECLARE_ALIGN(n) __attribute__((aligned(n)))

#endif /* PVBASE_TYPES_H */
