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

class PVDisplayViewTimeseries : public PVDisplayViewIf
{
  public:
	PVDisplayViewTimeseries();

  public:
	QWidget*
	create_widget(Inendi::PVView* view, QWidget* parent, Params const& data = {}) const override;
	bool abscissa_filter(Inendi::PVView* view, PVCol axis) const;
	void add_to_axis_menu(QMenu& menu,
	                      PVCol axis,
	                      PVCombCol axis_comb,
	                      Inendi::PVView*,
	                      PVDisplaysContainer* container) override;

	CLASS_REGISTRABLE(PVDisplayViewTimeseries)
};
} // namespace PVDisplays

#endif // PVDISPLAYS_PVDISPLAYVIEWTIMESERIES_H
