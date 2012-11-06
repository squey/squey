#include <picviz/PVSource.h> // Necesseray so that casting to PVCore::PVDataTreeObjectBase works!

#include <pvguiqt/PVRootTreeModel.h>
#include <pvguiqt/PVRootTreeView.h>

#include <pvguiqt/PVDisplaySourceDataTree.h>

PVDisplays::PVDisplaySourceDataTree::PVDisplaySourceDataTree():
	PVDisplaySourceIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::UniquePerParameters)
{
}

QWidget* PVDisplays::PVDisplaySourceDataTree::create_widget(Picviz::PVSource* src, QWidget* parent) const
{
	PVGuiQt::PVRootTreeModel* model  = new PVGuiQt::PVRootTreeModel(*src);
	PVGuiQt::PVRootTreeView*  widget = new PVGuiQt::PVRootTreeView(model, parent);

	return widget;
}
