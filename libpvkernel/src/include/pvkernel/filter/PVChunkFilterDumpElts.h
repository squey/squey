/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVCHUNKFILTERDUMPELTS_H
#define PVFILTER_PVCHUNKFILTERDUMPELTS_H

#include <pvkernel/filter/PVChunkFilter.h>

#include <map>

namespace PVFilter {

/**
 * This is a filter which doesn't change the PVChunk but save invalid elements
 * in the QStringList set at contruct time.
 */
class PVChunkFilterDumpElts : public PVChunkFilter {

public:
	PVChunkFilterDumpElts(std::map<size_t, std::string>& l);

	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);

protected:
	std::map<size_t, std::string>& _l; //!< List with invlaid elements.

	CLASS_FILTER_NONREG_NOPARAM(PVChunkFilterDumpElts)
};

}

#endif
