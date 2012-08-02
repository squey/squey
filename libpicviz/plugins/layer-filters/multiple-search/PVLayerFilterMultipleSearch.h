/**
 * \file PVLayerFilterMultipleSearch.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVLAYERFILTERMULTIPLESEARCH_H
#define PICVIZ_PVLAYERFILTERMULTIPLESEARCH_H

#include <pvkernel/core/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterMultipleSearch
 */
class PVLayerFilterMultipleSearch : public PVLayerFilter {
public:
	PVLayerFilterMultipleSearch(PVCore::PVArgumentList const& l = PVLayerFilterMultipleSearch::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);
	virtual QList<PVCore::PVArgumentKey> get_args_keys_for_preset() const;

	CLASS_FILTER(Picviz::PVLayerFilterMultipleSearch)
};

}

#endif
