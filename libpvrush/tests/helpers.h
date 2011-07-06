#ifndef HELPERS_VALID_FILE_H
#define HELPERS_VALID_FILE_H

#include <pvcore/PVChunk.h>
#include <pvcore/PVField.h>
#include <pvcore/PVElement.h>
#include <pvfilter/PVRawSourceBase.h>
#include <pvfilter/PVChunkFilter.h>
#include <pvrush/PVNraw.h>

void dump_chunk(PVCore::PVChunk const& c);
void dump_chunk_csv(PVCore::PVChunk&c);
void dump_chunk_raw(PVCore::PVChunk&c);
void dump_elt(PVCore::PVElement const& elt);
void dump_field(PVCore::PVField const& f);
void dump_buffer(char* start, char* end);
bool process_filter(PVFilter::PVRawSourceBase& source, PVFilter::PVChunkFilter_f flt_f);
void dump_nraw_csv(PVRush::PVNraw& nraw_);

#endif
