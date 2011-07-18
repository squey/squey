//! \file PVLayerFilterSearch.h
//! $Id: PVLayerFilterSearch.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLAYERFILTERSearch_H
#define PICVIZ_PVLAYERFILTERSearch_H


#include <pvcore/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterSearch
 */
class PVLayerFilterSearch : public PVLayerFilter {
public:
	PVLayerFilterSearch(PVFilter::PVArgumentList const& l = PVLayerFilterSearch::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);

	CLASS_FILTER(Picviz::PVLayerFilterSearch)

};
}

#endif
