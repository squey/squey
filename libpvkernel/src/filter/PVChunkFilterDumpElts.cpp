/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/filter/PVChunkFilterDumpElts.h>

/******************************************************************************
 *
 * PVFilter::PVChunkFilterDumpElts::PVChunkFilterDumpElts
 *
 *****************************************************************************/
PVFilter::PVChunkFilterDumpElts::PVChunkFilterDumpElts(QStringList& l):
	PVChunkFilter(), _l(l)
{
}

/******************************************************************************
 *
 * PVFilter::PVChunkFilterDumpElts::operator()
 *
 *****************************************************************************/
PVCore::PVChunk* PVFilter::PVChunkFilterDumpElts::operator()(PVCore::PVChunk* chunk)
{
	for (PVCore::PVElement* elt: chunk->elements()) {
		if (not elt->valid()) {
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

