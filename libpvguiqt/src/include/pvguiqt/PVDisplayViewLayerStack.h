#ifndef PVDISPLAYS_PVDISPLAYVIEWLAYERSTACK_H
#define PVDISPLAYS_PVDISPLAYVIEWLAYERSTACK_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays {

class PVDisplayViewLayerStack: public PVDisplayViewIf
{
public:
	PVDisplayViewLayerStack();

public:
	QWidget* create_widget(Picviz::PVView* view, QWidget* parent) const override;
	QIcon toolbar_icon() const override;
	QString widget_title(Picviz::PVView* view) const override;

	CLASS_REGISTRABLE(PVDisplayViewLayerStack)
};

}

#endif
