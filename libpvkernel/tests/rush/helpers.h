/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef HELPERS_VALID_FILE_H
#define HELPERS_VALID_FILE_H

#include <pvkernel/core/PVTextChunk.h>

void dump_chunk_csv(PVCore::PVTextChunk& c, std::ostream& out);
void dump_chunk_raw(PVCore::PVTextChunk const& c);

#endif
