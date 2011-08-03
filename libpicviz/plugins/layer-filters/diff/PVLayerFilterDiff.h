//! \file PVLayerFilterDiff.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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

		CLASS_FILTER(Picviz::PVLayerFilterDiff)
};

}

#endif
