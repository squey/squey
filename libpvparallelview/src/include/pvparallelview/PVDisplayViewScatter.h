/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVDISPLAYVIEWSCATTER_H__
#define __PVDISPLAYVIEWSCATTER_H__

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays
{

class PVDisplayViewScatter : public PVDisplayViewZoneIf
{
  public:
	PVDisplayViewScatter();

  public:
	QWidget* create_widget(Inendi::PVView* view,
	                       PVCombCol axis_comb_x,
	                       PVCombCol axis_comb_y,
	                       QWidget* parent) const override;
	QIcon toolbar_icon() const override;
	QString
	widget_title(Inendi::PVView* view, PVCombCol axis_comb_x, PVCombCol axis_comb_y) const override;
	QString axis_menu_name(Inendi::PVView const* view,
	                       PVCombCol axis_comb_x,
	                       PVCombCol axis_comb_y) const override;

	CLASS_REGISTRABLE(PVDisplayViewScatter)
};
} // namespace PVDisplays

#endif // __PVDISPLAYVIEWSCATTER_H__
