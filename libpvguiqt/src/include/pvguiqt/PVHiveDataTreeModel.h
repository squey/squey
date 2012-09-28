#ifndef DATATREEMODEL_H
#define DATATREEMODEL_H

#include <pvkernel/widgets/PVDataTreeModel.h>
#include <pvhive/PVObserverSignal.h>

#include <list>

namespace PVGuiQt {

class PVHiveDataTreeModel: public PVWidgets::PVDataTreeModel
{
	Q_OBJECT

	typedef PVHive::PVObserverSignal<PVCore::PVDataTreeObjectBase> datatree_obs_t;
public:
	PVHiveDataTreeModel(PVCore::PVDataTreeObjectBase& root, QObject* parent = 0);

private slots:
	void hive_refresh(PVHive::PVObserverBase* o);
	void about_to_be_deleted(PVHive::PVObserverBase* o);

private:
	std::list<datatree_obs_t> _obs;
};

}

#endif
