/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVLAYERFILTERERRORSSEARCH_H
#define INENDI_PVLAYERFILTERERRORSSEARCH_H

#include <inendi/PVLayer.h>
#include <inendi/PVLayerFilter.h>

#include <pvcop/db/exceptions/partially_converted_error.h>

namespace Inendi
{

/**
 * \class PVLayerFilterErrorsSearch
 */
class PVLayerFilterErrorsSearch : public PVLayerFilter
{

  public:
	PVLayerFilterErrorsSearch(
	    PVCore::PVArgumentList const& l = PVLayerFilterErrorsSearch::default_args());

  public:
	void operator()(PVLayer const& in, PVLayer& out) override;
	PVCore::PVArgumentKeyList get_args_keys_for_preset() const override;
	QString menu_name() const override { return "Text Search/Empty and invalid values"; }

  public:
	static PVCore::PVArgumentList menu(PVRow row, PVCol col, PVCol org_col, QString const& v);

	CLASS_FILTER(Inendi::PVLayerFilterErrorsSearch)
};
}

#endif
