/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef HELPERS_VALID_FILE_H
#define HELPERS_VALID_FILE_H

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/filter/PVChunkFilter.h>

void dump_chunk_csv(PVCore::PVChunk&c, std::ostream & out);
void dump_chunk_raw(PVCore::PVChunk const& c);

#endif
