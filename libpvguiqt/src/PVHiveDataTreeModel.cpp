#include <pvkernel/core/PVDataTreeObject.h>
#include <pvhive/PVHive.h>

#include <pvguiqt/PVHiveDataTreeModel.h>

PVGuiQt::PVHiveDataTreeModel::PVHiveDataTreeModel(PVCore::PVDataTreeObjectBase& root, QObject* parent):
	PVDataTreeModel(root, parent)
{
	// Register observers on the whole tree
	root.depth_first_list(
		[&](PVCore::PVDataTreeObjectBase* o) {
			this->_obs.emplace_back(static_cast<QObject*>(this));
			datatree_obs_t* obs = &_obs.back();
			auto datatree_o = o->base_shared_from_this();
			PVHive::get().register_observer(datatree_o, *obs);
			obs->connect_refresh(this, SLOT(hive_refresh(PVHive::PVObserverBase*)));
		}
	);
}

void PVGuiQt::PVHiveDataTreeModel::hive_refresh(PVHive::PVObserverBase* o)
{
	datatree_obs_t* real_o = dynamic_cast<datatree_obs_t*>(o);
	assert(real_o);
	const PVCore::PVDataTreeObjectBase* obj_base = real_o->get_object();

	// Find the index of this object
	QModelIndex idx = index_from_obj(obj_base);
	assert(idx.isValid());

	// Emit the fact that data has changed !
	emit dataChanged(idx, idx);
}
