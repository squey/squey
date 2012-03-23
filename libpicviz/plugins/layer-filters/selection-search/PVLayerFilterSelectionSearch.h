#ifndef PICVIZ_PVLAYERFILTERSELECTIONSEARCH_H
#define PICVIZ_PVLAYERFILTERSELECTIONSEARCH_H

#include <pvkernel/core/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterSelectionSearch
 */
class PVLayerFilterSelectionSearch : public PVLayerFilter {
public:
	PVLayerFilterSelectionSearch(PVCore::PVArgumentList const& l = PVLayerFilterSelectionSearch::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);
	virtual QList<PVCore::PVArgumentKey> get_args_keys_for_preset() const;

public:
	static PVCore::PVArgumentList sel_axis_menu(PVRow /*row*/, PVCol col, QString const& /*v*/);

	CLASS_FILTER(Picviz::PVLayerFilterSelectionSearch)
};

}

#endif
