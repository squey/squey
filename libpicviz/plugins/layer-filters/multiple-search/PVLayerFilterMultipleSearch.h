/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
	virtual PVCore::PVArgumentKeyList get_args_keys_for_preset() const;
	virtual QString menu_name() const { return "Text Search/Multiple values"; }

public:
	static PVCore::PVArgumentList search_value_menu(PVRow row, PVCol col, PVCol org_col, QString const& v);
	static PVCore::PVArgumentList search_using_value_menu(PVRow row, PVCol col, PVCol org_col, QString const& v);
	static PVCore::PVArgumentList search_menu(PVRow row, PVCol col, PVCol org_col, QString const& v);

	CLASS_FILTER(Picviz::PVLayerFilterMultipleSearch)
};

}

#endif
