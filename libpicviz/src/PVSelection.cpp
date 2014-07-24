/**
 * \file PVSelection.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include "bithacks.h"

#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/PVBitCount.h>
#include <pvkernel/core/PVHardwareConcurrency.h>
#include <pvkernel/rush/PVNraw.h>

#include <picviz/PVSelection.h>
#include <picviz/PVSparseSelection.h>
#include <picviz/PVAxesCombination.h>

Picviz::PVSelection & Picviz::PVSelection::operator|=(const PVSparseSelection &rhs)
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
