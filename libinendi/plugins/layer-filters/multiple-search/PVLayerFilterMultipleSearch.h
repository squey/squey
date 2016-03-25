/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVLAYERFILTERMULTIPLESEARCH_H
#define INENDI_PVLAYERFILTERMULTIPLESEARCH_H

#include <pvkernel/core/general.h>

#include <inendi/PVLayer.h>
#include <inendi/PVLayerFilter.h>

namespace Inendi {

/**
 * \class PVLayerFilterMultipleSearch
 */
class PVLayerFilterMultipleSearch : public PVLayerFilter {

public:
	PVLayerFilterMultipleSearch(PVCore::PVArgumentList const& l = PVLayerFilterMultipleSearch::default_args());
public:
	void operator()(PVLayer const& in, PVLayer &out) override;
	PVCore::PVArgumentKeyList get_args_keys_for_preset() const override;
	QString menu_name() const override { return "Text Search/Multiple values"; }

public:
	static PVCore::PVArgumentList search_value_menu(PVRow row, PVCol col, PVCol org_col, QString const& v);
	static PVCore::PVArgumentList search_using_value_menu(PVRow row, PVCol col, PVCol org_col, QString const& v);
	static PVCore::PVArgumentList search_menu(PVRow row, PVCol col, PVCol org_col, QString const& v);

	CLASS_FILTER(Inendi::PVLayerFilterMultipleSearch)
};

}

#endif
