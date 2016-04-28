/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVLAYERFILTERAXISGRADIENT_H
#define INENDI_PVLAYERFILTERAXISGRADIENT_H

#include <pvkernel/core/general.h>

#include <inendi/PVLayer.h>
#include <inendi/PVLayerFilter.h>

namespace Inendi
{

/**
 * \class PVLayerFilterAxisGradient
 */
class PVLayerFilterAxisGradient : public PVLayerFilter
{
  public:
	PVLayerFilterAxisGradient(
	    PVCore::PVArgumentList const& l = PVLayerFilterAxisGradient::default_args());

  public:
	void operator()(PVLayer const& in, PVLayer& out) override;
	PVCore::PVArgumentKeyList get_args_keys_for_preset() const override;
	QString status_bar_description() override;
	QString detailed_description() override;
	QString menu_name() const override { return "Axis gradient"; }

  public:
	static PVCore::PVArgumentList
	gradient_menu(PVRow row, PVCol col, PVCol org_col, QString const& v);

	CLASS_FILTER(Inendi::PVLayerFilterAxisGradient)
};
}

#endif
