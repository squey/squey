//! \file PVChunkFilterDumpElts.cpp
//! $Id: PVChunkFilterDumpElts.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <pvfilter/PVChunkFilterDumpElts.h>
#include <assert.h>

/******************************************************************************
 *
 * PVFilter::PVChunkFilterDumpElts::PVChunkFilterDumpElts
 *
 *****************************************************************************/
PVFilter::PVChunkFilterDumpElts::PVChunkFilterDumpElts(bool dump_valid, QStringList& l):
	PVChunkFilter(), _dump_valid(dump_valid), _l(l)
{
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilterDumpElts::operator()
 *
 *****************************************************************************/
PVCore::PVChunk* PVFilter::PVChunkFilterDumpElts::operator()(PVCore::PVChunk* chunk)
{
	PVCore::list_elts::iterator it,ite;
	PVCore::list_elts& elts = chunk->elements();
	ite = elts.end();
	for (it = elts.begin(); it != ite; it++) {
		bool bValid = it->valid();
		if ((bValid && _dump_valid) || (!bValid && !_dump_valid)) {
			it->init_qstr();
			QString deep_copy(it->qstr().unicode(), it->qstr().size());
			_l << deep_copy;
		}
	}

	return chunk;
}

