/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVBASE_TYPES_H
#define PVBASE_TYPES_H

#include <QtGlobal>

typedef qint32 PVCol;
typedef quint32 PVRow;

#define PVROW_INVALID_VALUE 0xFFFFFFFF
#define PVCOL_INVALID_VALUE ((PVCol)-1)

#define PVROW_VECTOR_ALIGNEMENT                                                                    \
	(128 / (sizeof(PVRow) * 8)) // Define necessary alignement of pointers of
                                // PVRows for vectorisation usage

using chunk_index = quint64;

#define DECLARE_ALIGN(n) __attribute__((aligned(n)))

#endif /* PVBASE_TYPES_H */
