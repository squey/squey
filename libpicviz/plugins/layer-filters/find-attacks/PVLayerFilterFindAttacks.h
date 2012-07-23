/**
 * \file PVLayerFilterFindAttacks.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVLAYERFILTERFindAttacks_H
#define PICVIZ_PVLAYERFILTERFindAttacks_H

#include <pvkernel/core/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterFindAttacks
 */
class PVLayerFilterFindAttacks : public PVLayerFilter {
public:
	PVLayerFilterFindAttacks(PVCore::PVArgumentList const& l = PVLayerFilterFindAttacks::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);

	CLASS_FILTER(Picviz::PVLayerFilterFindAttacks)

};
}

#endif
