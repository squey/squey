/**
 * \file PVLayerFilterSearch.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVLAYERFILTERSearch_H
#define PICVIZ_PVLAYERFILTERSearch_H


#include <pvkernel/core/general.h>

#include <picviz/PVLayer.h>
#include <picviz/PVLayerFilter.h>

namespace Picviz {

/**
 * \class PVLayerFilterSearch
 */
class PVLayerFilterSearch : public PVLayerFilter {
public:
	PVLayerFilterSearch(PVCore::PVArgumentList const& l = PVLayerFilterSearch::default_args());
public:
	virtual void operator()(PVLayer& in, PVLayer &out);
	virtual QList<PVCore::PVArgumentKey> get_args_keys_for_preset() const;

public:
	static PVCore::PVArgumentList search_value_menu(PVRow row, PVCol col, QString const& v);

	CLASS_FILTER(Picviz::PVLayerFilterSearch)

};
}

#endif
