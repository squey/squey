/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVBASE_QHASHES_H
#define PVBASE_QHASHES_H

// Defines qhash overrides
// They need to be included before QHash !
#ifdef QHASH_H
#error pvbase/qhashes.h must be included before QHash !
#endif

#include <pvkernel/core/string_tbb.h>

unsigned int qHash(std::string_tbb const& str);

#endif
