/**
 * \file PVChunkFilterDumpElts.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <pvkernel/filter/PVChunkFilterDumpElts.h>
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
		PVCore::PVElement* elt = *it;
		bool bValid = elt->valid();
		if (bValid && _dump_valid) {
			QString deep_copy((const QChar*) elt->begin(), elt->size()/sizeof(QChar));
			_l << deep_copy;
		}
		else
		if (!bValid && !_dump_valid) {
			size_t saved_buf_size = 0;
			char* saved_buf = elt->get_saved_elt_buffer(saved_buf_size);
			if (saved_buf) {
				QString str_elt((QChar*) saved_buf, saved_buf_size/sizeof(QChar));
				_l << str_elt;
			}
			else {
				//PVLOG_WARN("(PVChunkFilterDumpElts) WARNING: no copy of the original element exists. The value saved for an invalid element might be completely changed by previous filters... Remember to use PVChunkFilterByEltSaveInvalid or PVChunkFilterByEltRestoreInvalid to avoid this issue !\n");
				QString deep_copy((const QChar*) elt->begin(), elt->size()/sizeof(QChar));
				_l << deep_copy;
			}
		}
	}

	return chunk;
}

