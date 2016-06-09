/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_PVROOTREEMODEL_H
#define PVGUIQT_PVROOTREEMODEL_H

#include <pvguiqt/PVHiveDataTreeModel.h>

namespace PVGuiQt
{

class PVRootTreeModel : public PVHiveDataTreeModel
{
	Q_OBJECT

  public:
	PVRootTreeModel(Inendi::PVSource& root, QObject* parent = 0);

  public:
	QVariant data(const QModelIndex& index, int role) const;
};
}

#endif
