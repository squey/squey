/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "bithacks.h"

#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/core/PVBitCount.h>
#include <pvkernel/core/PVHardwareConcurrency.h>
#include <pvkernel/rush/PVNraw.h>

#include <inendi/PVSelection.h>
#include <inendi/PVSparseSelection.h>
#include <inendi/PVAxesCombination.h>

Inendi::PVSelection& Inendi::PVSelection::operator|=(const PVSparseSelection& rhs)
{
	if (!_table) {
		allocate_table();
	}

	PVSparseSelection::map_chunks_t const& chunks = rhs.get_chunks();
	PVSparseSelection::map_chunks_t::const_iterator it;
	for (it = chunks.begin(); it != chunks.end(); it++) {
		_table[it->first] |= it->second;
	}

	return *this;
}
