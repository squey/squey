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

void Picviz::PVSelection::write_selected_lines_nraw(QTextStream& stream, PVRush::PVNraw const& nraw,
                                                    PVRow start, PVRow count)
{
	if (!_table) {
		return;
	}
	PVRow nrows = nraw.get_number_rows();
	assert(nrows > 0);
#ifndef NDEBUG
	PVCol ncols = nraw.get_number_cols();
	assert(ncols > 0);
#endif

	PVRow nrows_counter = 0;

	for (PVRow line_index = start; line_index < nrows; line_index++) {
		if (!get_line_fast(line_index)) {
			continue;
		}

		if (nrows_counter == count) {
			return;
		}

		QString line = nraw.nraw_line_to_csv(line_index);
		stream << line << QString("\n");
		nrows_counter++;
	}
}
