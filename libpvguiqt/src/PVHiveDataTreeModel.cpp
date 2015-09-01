#include <pvkernel/core/PVDataTreeObject.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVObserverCallback.h>

#include <pvguiqt/PVHiveDataTreeModel.h>

PVGuiQt::PVHiveDataTreeModel::PVHiveDataTreeModel(PVCore::PVDataTreeObjectBase& root, QObject* parent):
	PVDataTreeModel(root, parent)
{
	register_all_observers();

	// Same with root!
	this->_obs.emplace_back(static_cast<QObject*>(this));
	datatree_obs_t* obs = &_obs.back();
	auto datatree_o = root.base_shared_from_this();
	PVHive::get().register_observer(datatree_o, *obs);
	obs->connect_refresh(this, SLOT(hive_refresh(PVHive::PVObserverBase*)));
	obs->connect_about_to_be_deleted(this, SLOT(root_about_to_be_deleted(PVHive::PVObserverBase*)));

	_root_recursive_observer = PVHive::create_observer_callback_heap<PVCore::PVDataTreeObjectBase>(
		[](PVCore::PVDataTreeObjectBase const*) { },
		[&](PVCore::PVDataTreeObjectBase const*) { register_all_observers(); },
		[](PVCore::PVDataTreeObjectBase const*) { });
	_root_recursive_observer->set_accept_recursive_refreshes(true);

	auto root_sp = root.base_shared_from_this();
	PVHive::get().register_observer(root_sp, *_root_recursive_observer);
}

int PVGuiQt::PVHiveDataTreeModel::rowCount(const QModelIndex &index) const
{
	if (!_view_valid) {
		return 0;
	}

	return PVWidgets::PVDataTreeModel::rowCount(index);
}

bool PVGuiQt::PVHiveDataTreeModel::is_object_observed(PVCore::PVDataTreeObjectBase* o) const
{
	for (datatree_obs_t const& obs: _obs) {
		if (obs.get_object() == o) {
			return true;
		}
	}
	return false;
}

void PVGuiQt::PVHiveDataTreeModel::register_all_observers()
{
	// Register observers on the whole tree
	_root_base->depth_first_list(
		[&](PVCore::PVDataTreeObjectBase* o) {
			if (!is_object_observed(o)) {
				this->_obs.emplace_back(static_cast<QObject*>(this));
				datatree_obs_t* obs = &_obs.back();
				auto datatree_o = o->base_shared_from_this();
				PVHive::get().register_observer(datatree_o, *obs);
				obs->connect_refresh(this, SLOT(hive_refresh(PVHive::PVObserverBase*)));
				obs->connect_about_to_be_deleted(this, SLOT(about_to_be_deleted(PVHive::PVObserverBase*)));

				// Refresh parent
				PVCore::PVDataTreeObjectWithParentBase* o_with_parent = o->cast_with_parent();
				if (o_with_parent) {
					/*QModelIndex idx = index_from_obj(o_with_parent->get_parent_base());
					emit dataChanged(idx, idx);*/
					beginResetModel();
					endResetModel();
				}
			}
		}
	);
}

void PVGuiQt::PVHiveDataTreeModel::hive_refresh(PVHive::PVObserverBase* o)
{
	datatree_obs_t* real_o = dynamic_cast<datatree_obs_t*>(o);
	assert(real_o);
	const PVCore::PVDataTreeObjectBase* obj_base = real_o->get_object();
	if (dynamic_cast<PVCore::PVDataTreeObjectWithChildrenBase const*>(obj_base) == _root) {
		beginResetModel();
		endResetModel();
		return;
	}

	// Find the index of this object
	QModelIndex idx = index_from_obj(obj_base);
	assert(idx.isValid());

	// Emit the fact that data has changed !
	emit dataChanged(idx, idx);
}

void PVGuiQt::PVHiveDataTreeModel::root_about_to_be_deleted(PVHive::PVObserverBase*)
{
	beginResetModel();
	_view_valid = false;
	endResetModel();
}

void PVGuiQt::PVHiveDataTreeModel::about_to_be_deleted(PVHive::PVObserverBase*)
{
	/*datatree_obs_t* real_o = dynamic_cast<datatree_obs_t*>(o);
	assert(real_o);
	const PVCore::PVDataTreeObjectBase* obj_base = real_o->get_object();
	QModelIndex idx = index_from_obj(obj_base);

	if (idx.isValid()) {
		removeRows(idx.row(), idx.column(), idx.parent());
	}*/
	beginResetModel();
	endResetModel();
}
