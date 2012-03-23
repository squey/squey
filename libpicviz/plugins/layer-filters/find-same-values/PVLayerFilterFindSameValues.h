//! \file PVLayerFilterFindSameValues.h
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
 * \class PVLayerFilterFindSameValues
 */
class PVLayerFilterFindSameValues : public PVLayerFilter {
	private:
		PVCore::PVArgumentList get_default_args_for_view(PVView const& view);
	public:
		PVLayerFilterFindSameValues(PVCore::PVArgumentList const& l = PVLayerFilterFindSameValues::default_args());	

		virtual void operator()(PVLayer& in, PVLayer &out);
		virtual QList<PVCore::PVArgumentKey> get_args_keys_for_preset() const;

		CLASS_FILTER(Picviz::PVLayerFilterFindSameValues)
};

}

#endif
