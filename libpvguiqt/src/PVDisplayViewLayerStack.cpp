#include <pvguiqt/PVLayerStackDelegate.h>
#include <pvguiqt/PVLayerStackModel.h>
#include <pvguiqt/PVLayerStackView.h>

#include <pvguiqt/PVDisplayViewLayerStack.h>

PVDisplays::PVDisplayViewLayerStack::PVDisplayViewLayerStack():
	PVDisplayViewIf(PVDisplayIf::ShowInToolbar)
{
}

QWidget* PVDisplays::PVDisplayViewLayerStack::create_widget(Picviz::PVView* view, QWidget* parent) const
{
	Picviz::PVView_sp view_sp = view->shared_from_this();

	PVGuiQt::PVLayerStackDelegate* delegate = new PVGuiQt::PVLayerStackDelegate(*view);
	PVGuiQt::PVLayerStackModel* model  = new PVGuiQt::PVLayerStackModel(view_sp);
	PVGuiQt::PVLayerStackView*  widget = new PVGuiQt::PVLayerStackView(parent);
	widget->setModel(model);
	widget->setItemDelegate(delegate);

	return widget;
}
