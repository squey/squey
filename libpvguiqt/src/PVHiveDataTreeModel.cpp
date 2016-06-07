/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */
#include <pvhive/PVObserverCallback.h>
#include <pvguiqt/PVHiveDataTreeModel.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVMapped.h>
#include <inendi/PVSource.h>

#include <list>

namespace PVGuiQt
{

PVHiveDataTreeModel::PVHiveDataTreeModel(Inendi::PVSource& root, QObject* parent)
    : QAbstractItemModel(parent), _root(root)
{
	register_all_observers();

	register_obs(&root);

	_root_recursive_observer = PVHive::create_observer_callback_heap<PVCore::PVDataTreeObject>(
	    [](PVCore::PVDataTreeObject const*) {},
	    [this](PVCore::PVDataTreeObject const*) { register_all_observers(); },
	    [](PVCore::PVDataTreeObject const*) {});
	_root_recursive_observer->set_accept_recursive_refreshes(true);

	auto root_sp = root.shared_from_this();
	PVHive::get().register_observer(root_sp, *_root_recursive_observer);
}

QModelIndex PVHiveDataTreeModel::index(int row, int column, const QModelIndex& parent) const
{
	// Column is always 0 (see columnCount), but asserts it
	assert(column == 0);

	PVCore::PVDataTreeObject* p =
	    (parent.isValid()) ? (PVCore::PVDataTreeObject*)parent.internalPointer() : &_root;

	PVCore::PVDataTreeObject* child = nullptr;
	if (Inendi::PVPlotted* v = dynamic_cast<Inendi::PVPlotted*>(p)) {
		auto children = v->get_children();
		auto it = children.begin();
		std::advance(it, row);
		child = it->get();
	} else if (Inendi::PVMapped* v = dynamic_cast<Inendi::PVMapped*>(p)) {
		auto children = v->get_children();
		auto it = children.begin();
		std::advance(it, row);
		child = it->get();
	} else if (Inendi::PVSource* v = dynamic_cast<Inendi::PVSource*>(p)) {
		auto children = v->get_children();
		auto it = children.begin();
		std::advance(it, row);
		child = it->get();
	} else {
		throw std::runtime_error("Invalid kind of node");
	}

	return createIndex(row, column, child);
}

int PVHiveDataTreeModel::rowCount(const QModelIndex& parent) const
{
	if (not parent.isValid()) {
		return 1;
	} else if (Inendi::PVPlotted* v = dynamic_cast<Inendi::PVPlotted*>(
	               (PVCore::PVDataTreeObject*)parent.internalPointer())) {
		return v->size();
	} else if (Inendi::PVMapped* v = dynamic_cast<Inendi::PVMapped*>(
	               (PVCore::PVDataTreeObject*)parent.internalPointer())) {
		return v->size();
	} else if (Inendi::PVSource* v = dynamic_cast<Inendi::PVSource*>(
	               (PVCore::PVDataTreeObject*)parent.internalPointer())) {
		return v->size();
	} else {
		return 0; // View case
	}
}

QModelIndex PVHiveDataTreeModel::parent(const QModelIndex& index) const
{
	if (!index.isValid()) {
		return {};
	}

	if (index.internalPointer() == &_root) {
		return {};
	}

	PVCore::PVDataTreeObject* parent = nullptr;
	PVCore::PVDataTreeObject* id = (PVCore::PVDataTreeObject*)index.internalPointer();
	int row = 0;

	if (Inendi::PVPlotted* v = dynamic_cast<Inendi::PVPlotted*>(id)) {
		parent = v->get_parent();
		row = pos_from_obj(v->get_parent());
	} else if (Inendi::PVMapped* v = dynamic_cast<Inendi::PVMapped*>(id)) {
		parent = v->get_parent();
		row = 0;
	} else if (Inendi::PVView* v = dynamic_cast<Inendi::PVView*>(id)) {
		parent = v->get_parent();
		row = pos_from_obj(v->get_parent());
	} else {
		throw std::runtime_error("Invalid kind of node asked for parent");
	}

	return createIndex(row, 0, parent);
}

QVariant PVHiveDataTreeModel::data(const QModelIndex& index, int role) const
{

	PVCore::PVDataTreeObject* ptr;
	if (!index.isValid()) {
		ptr = &_root;
	} else {
		ptr = (PVCore::PVDataTreeObject*)index.internalPointer();
	}

	if (role == Qt::DisplayRole) {
		return QString::fromStdString(ptr->get_serialize_description());
	}

	return {};
}

void PVHiveDataTreeModel::hive_refresh(PVHive::PVObserverBase* o)
{
	datatree_obs_t* real_o = dynamic_cast<datatree_obs_t*>(o);
	assert(real_o);
	const PVCore::PVDataTreeObject* obj_base = real_o->get_object();
	if (obj_base == &_root) {
		beginResetModel();
		endResetModel();
		return;
	}

	// Find the index of this object
	QModelIndex idx = createIndex(pos_from_obj(obj_base), 0, (void*)obj_base);
	assert(idx.isValid());

	// Emit the fact that data has changed !
	emit dataChanged(idx, idx);
}

int PVHiveDataTreeModel::pos_from_obj(PVCore::PVDataTreeObject const* id) const
{
	if (Inendi::PVPlotted const* v = dynamic_cast<Inendi::PVPlotted const*>(id)) {
		auto children = v->get_parent()->get_children();
		return std::distance(
		    children.begin(),
		    std::find_if(children.begin(), children.end(),
		                 [v](PVCore::PVSharedPtr<const Inendi::PVPlotted> const& n) {
			                 return n.get() == v;
			             }));
	} else if (Inendi::PVMapped const* v = dynamic_cast<Inendi::PVMapped const*>(id)) {
		auto children = v->get_parent()->get_children();
		return std::distance(
		    children.begin(),
		    std::find_if(children.begin(), children.end(),
		                 [v](PVCore::PVSharedPtr<const Inendi::PVMapped> const& n) {
			                 return n.get() == v;
			             }));
	} else if (Inendi::PVView const* v = dynamic_cast<Inendi::PVView const*>(id)) {
		auto children = v->get_parent()->get_children();
		return std::distance(children.begin(),
		                     std::find_if(children.begin(), children.end(),
		                                  [v](PVCore::PVSharedPtr<const Inendi::PVView> const& n) {
			                                  return n.get() == v;
			                              }));
	} else if (dynamic_cast<Inendi::PVSource const*>(id)) {
		return 0;
	} else {
		throw std::runtime_error("Invalid kind of node asked for parent");
	}
}
}
