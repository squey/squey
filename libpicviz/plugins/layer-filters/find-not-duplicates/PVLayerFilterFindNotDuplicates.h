//! \file PVLayerFilterFindNotDuplicates.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLAYERFILTERFindNotDuplicates_H
#define PICVIZ_PVLAYERFILTERFindNotDuplicates_H


#include <pvkernel/core/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterFindNotDuplicates
 */
class PVLayerFilterFindNotDuplicates : public PVLayerFilter {
public:
	PVLayerFilterFindNotDuplicates(PVCore::PVArgumentList const& l = PVLayerFilterFindNotDuplicates::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);

	CLASS_FILTER(Picviz::PVLayerFilterFindNotDuplicates)
};
}

#endif
