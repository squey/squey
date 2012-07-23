/**
 * \file PVLayerFilterSearchPlotOneToMany.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVLAYERFILTERSearchPlotOneToMany_H
#define PICVIZ_PVLAYERFILTERSearchPlotOneToMany_H


#include <pvkernel/core/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterSearchPlotOneToMany
 */
class PVLayerFilterSearchPlotOneToMany : public PVLayerFilter {
public:
	PVLayerFilterSearchPlotOneToMany(PVCore::PVArgumentList const& l = PVLayerFilterSearchPlotOneToMany::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);
	virtual QList<PVCore::PVArgumentKey> get_args_keys_for_preset() const;
	virtual QString menu_name() const { return "Plot Search/One to many"; }

public:
	CLASS_FILTER(Picviz::PVLayerFilterSearchPlotOneToMany)

};
}

#endif	/* PVLayerFilterSearchPlotOneToMany */
