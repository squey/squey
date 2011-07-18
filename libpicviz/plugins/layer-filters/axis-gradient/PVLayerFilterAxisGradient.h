//! \file PVLayerFilterAxisGradient.h
//! $Id: PVLayerFilterAxisGradient.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLAYERFILTERAXISGRADIENT_H
#define PICVIZ_PVLAYERFILTERAXISGRADIENT_H


#include <pvcore/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterAxisGradient
 */
class PVLayerFilterAxisGradient : public PVLayerFilter {
public:
	PVLayerFilterAxisGradient(PVFilter::PVArgumentList const& l = PVLayerFilterAxisGradient::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);
	virtual QString status_bar_description();
	virtual QString detailed_description();

	CLASS_FILTER(Picviz::PVLayerFilterAxisGradient)

};
}

#endif
