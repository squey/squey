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
PVFilter::PVChunkFilterDumpElts::PVChunkFilterDumpElts(std::map<size_t, std::string>& l):
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
	// TODO : This should be const
	for (PVCore::PVElement* elt: chunk->elements()) {
		if (not elt->valid()) {
			// TODO : Could be remove when UTF16 is remove too.
			QString deep_copy((const QChar*) elt->begin(), elt->size()/sizeof(QChar));
			_l[elt->get_elt_agg_index()] = deep_copy.toStdString();
		}
	}

	return chunk;
}

