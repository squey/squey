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

class PVDisplayViewScatter : public PVDisplayViewIf
{
  public:
	PVDisplayViewScatter();

  public:
	QWidget*
	create_widget(Inendi::PVView* view, QWidget* parent, Params const& data = {}) const override;
	QIcon toolbar_icon() const override;
	QString widget_title(Inendi::PVView* view) const override;
	QString axis_menu_name(Inendi::PVView* view) const override;
	void add_to_axis_menu(QMenu& menu,
	                      PVCombCol axis_comb,
	                      Inendi::PVView*,
	                      PVDisplaysContainer* container) override;

	CLASS_REGISTRABLE(PVDisplayViewScatter)
};
} // namespace PVDisplays

#endif // __PVDISPLAYVIEWSCATTER_H__
