//! \file PVLayerFilterFindDuplicates.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLAYERFILTERFindDuplicates_H
#define PICVIZ_PVLAYERFILTERFindDuplicates_H


#include <pvkernel/core/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterFindDuplicates
 */
class PVLayerFilterFindDuplicates : public PVLayerFilter {
public:
	PVLayerFilterFindDuplicates(PVCore::PVArgumentList const& l = PVLayerFilterFindDuplicates::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);

public:
	CLASS_FILTER(Picviz::PVLayerFilterFindDuplicates)
};
}

#endif
