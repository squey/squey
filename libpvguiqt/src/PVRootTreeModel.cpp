#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include <pvguiqt/PVRootTreeModel.h>

#include <QFont>

PVGuiQt::PVRootTreeModel::PVRootTreeModel(PVCore::PVDataTreeObjectBase& root, QObject* parent):
	PVHiveDataTreeModel(root, parent)
{
}

QVariant PVGuiQt::PVRootTreeModel::data(const QModelIndex& index, int role) const
{
	if (role == Qt::FontRole) {
		PVCore::PVDataTreeObjectBase const* obj = (PVCore::PVDataTreeObjectBase const*) index.internalPointer();
		Picviz::PVView const* view = dynamic_cast<Picviz::PVView const*>(obj);
		if (view && view->get_parent<Picviz::PVSource>()->current_view() == view) {
		   QFont font;
		   font.setBold(true);
		   return font;
		}
	}

	return PVDataTreeModel::data(index, role);
}