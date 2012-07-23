/**
 * \file PVLayerFilterDiff.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVLAYERFILTERDiff_H
#define PICVIZ_PVLAYERFILTERDiff_H

#include <pvkernel/core/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>
#include <picviz/PVView.h>

namespace Picviz {

/**
 * \class PVLayerFilterDiff
 */
class PVLayerFilterDiff : public PVLayerFilter {
private:
	PVCore::PVArgumentList get_default_args_for_view(PVView const& view);
public:
	PVLayerFilterDiff(PVCore::PVArgumentList const& l = PVLayerFilterDiff::default_args());	

	virtual void operator()(PVLayer& in, PVLayer &out);
	virtual QList<PVCore::PVArgumentKey> get_args_keys_for_preset() const;
	virtual QString menu_name() const { return "Diff"; }

	CLASS_FILTER(Picviz::PVLayerFilterDiff)
};

}

#endif
