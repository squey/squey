/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#ifndef PVDISPLAYS_PVDISPLAYVIEWTIMESERIES_H
#define PVDISPLAYS_PVDISPLAYVIEWTIMESERIES_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays
{

class PVDisplayViewTimeseries : public PVDisplayViewAxisIf
{
  public:
	PVDisplayViewTimeseries();

  public:
	QWidget*
	create_widget(Inendi::PVView* view, PVCombCol axis_comb, QWidget* parent) const override;
	QIcon toolbar_icon() const override;
	QString widget_title(Inendi::PVView* view, PVCombCol axis_comb) const override;
	QString axis_menu_name(Inendi::PVView const* view, PVCombCol axis_comb) const override;
	bool should_add_to_menu(Inendi::PVView const* view, PVCombCol axis_comb) const override;

	CLASS_REGISTRABLE(PVDisplayViewTimeseries)
};
} // namespace PVDisplays

#endif // PVDISPLAYS_PVDISPLAYVIEWTIMESERIES_H
