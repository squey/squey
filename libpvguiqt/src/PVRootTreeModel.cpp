/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVRoot.h>
#include <inendi/PVView.h>

#include <pvguiqt/PVRootTreeModel.h>

#include <QBrush>
#include <QFont>

PVGuiQt::PVRootTreeModel::PVRootTreeModel(Inendi::PVSource& root, QObject* parent)
    : PVHiveDataTreeModel(root, parent)
{
}

QVariant PVGuiQt::PVRootTreeModel::data(const QModelIndex& index, int role) const
{
	if (Inendi::PVView* v =
	        dynamic_cast<Inendi::PVView*>((PVCore::PVDataTreeObject*)index.internalPointer())) {
		if (role == Qt::FontRole) {
			if (v->get_parent<Inendi::PVRoot>()->current_view() == v) {
				QFont font;
				font.setBold(true);
				return font;
			}
		} else if (role == Qt::ForegroundRole) {
			return QBrush(v->get_color());
		}
	}

	return PVHiveDataTreeModel::data(index, role);
}
