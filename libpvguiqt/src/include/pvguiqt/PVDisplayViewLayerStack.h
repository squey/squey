/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDISPLAYS_PVDISPLAYVIEWLAYERSTACK_H
#define PVDISPLAYS_PVDISPLAYVIEWLAYERSTACK_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays
{

class PVDisplayViewLayerStack : public PVDisplayViewIf
{
  public:
	PVDisplayViewLayerStack();

  public:
	QWidget*
	create_widget(Inendi::PVView* view, QWidget* parent, Params const& data = {}) const override;
	QIcon toolbar_icon() const override;
	QString widget_title(Inendi::PVView* view) const override;

	CLASS_REGISTRABLE(PVDisplayViewLayerStack)
};
} // namespace PVDisplays

#endif
