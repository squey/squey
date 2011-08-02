//! \file PVChunkFilterDumpElts.h
//! $Id: PVChunkFilterDumpElts.h 3187 2011-06-21 11:20:33Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

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
	virtual PVCore::PVChunk* operator()(PVCore::PVChunk* chunk);

protected:
	bool _dump_valid;
	QStringList& _l;
};

}

#endif
