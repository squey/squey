//! \file PVLayerFilterWebmailFinder.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVLAYERFILTERWebmailFinder_H
#define PICVIZ_PVLAYERFILTERWebmailFinder_H

#include <pvkernel/core/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterWebmailFinder
 */
class PVLayerFilterWebmailFinder : public PVLayerFilter {
public:
	PVLayerFilterWebmailFinder(PVCore::PVArgumentList const& l = PVLayerFilterWebmailFinder::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);
	PVCore::PVArgumentList get_default_args_for_view(PVView const& view);

	CLASS_FILTER(Picviz::PVLayerFilterWebmailFinder)

};
}

#endif
