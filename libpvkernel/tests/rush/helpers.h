/**
 * \file helpers.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef HELPERS_VALID_FILE_H
#define HELPERS_VALID_FILE_H

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/rush/PVNraw.h>

#include <QString>

void dump_chunk(PVCore::PVChunk const& c);
void dump_chunk_csv(PVCore::PVChunk&c);
void dump_chunk_raw(PVCore::PVChunk const& c);
void dump_chunk_newline(PVCore::PVChunk const& c);
void dump_elt(PVCore::PVElement const& elt);
void dump_field(PVCore::PVField const& f);
void dump_buffer(char* start, char* end);
bool process_filter(PVRush::PVRawSourceBase& source, PVFilter::PVChunkFilter_f flt_f);
void dump_nraw_csv(PVRush::PVNraw& nraw_);
void dump_nraw_csv(PVRush::PVNraw& nraw_, const QString& csv_path);

#endif
