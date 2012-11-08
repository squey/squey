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
	void register_all_observers();
	bool is_object_observed(PVCore::PVDataTreeObjectBase* o) const;

private:
	std::list<datatree_obs_t> _obs;
	PVHive::PVObserver_p<PVCore::PVDataTreeObjectBase> _root_recursive_observer;
};

}

#endif
