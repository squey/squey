/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVCHUNKFILTERDUMPELTS_H
#define PVFILTER_PVCHUNKFILTERDUMPELTS_H

#include <pvkernel/core/general.h>
#include <pvbase/types.h>
#include <pvkernel/filter/PVChunkFilter.h>

#include <QStringList>
#include <map>

namespace PVFilter {

class PVChunkFilterDumpElts : public PVChunkFilter {

public:
	PVChunkFilterDumpElts(bool dump_valid, QStringList& l);

public:
	PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);

protected:
	bool _dump_valid;
	QStringList& _l;

	CLASS_FILTER_NONREG_NOPARAM(PVChunkFilterDumpElts)
};

}

#endif
