/**
 * \file PVChunkFilterDumpElts.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVCHUNKFILTERDUMPELTS_H
#define PVFILTER_PVCHUNKFILTERDUMPELTS_H

#include <pvkernel/core/general.h>
#include <pvbase/types.h>
#include <pvkernel/filter/PVChunkFilter.h>

#include <QStringList>
#include <map>

namespace PVFilter {

class LibKernelDecl PVChunkFilterDumpElts : public PVChunkFilter {

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
