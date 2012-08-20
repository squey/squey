#ifndef PVGUIQT_PVROOTREEMODEL_H
#define PVGUIQT_PVROOTREEMODEL_H

#include <pvguiqt/PVHiveDataTreeModel.h>

namespace PVGuiQt {

class PVRootTreeModel: public PVHiveDataTreeModel
{
public:
	PVRootTreeModel(PVCore::PVDataTreeObjectBase& root, QObject* parent = 0);

public:
	QVariant data(const QModelIndex &index, int role) const;
};

}

#endif
