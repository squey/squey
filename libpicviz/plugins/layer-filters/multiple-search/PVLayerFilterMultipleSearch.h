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

	CLASS_FILTER(Picviz::PVLayerFilterMultipleSearch)
};

}

#endif
