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

/**
 * Helper to call a function on the first correct type (from dynamic cast check).
 *
 * @note : It is an O(n) call but we don't require performances in this widget.
 */
namespace __impl
{
template <class R, class F, class In, size_t I, class... T>
typename std::enable_if<I == sizeof...(T), R>::type apply_on(In*, F const&)
{
	throw std::runtime_error("No possible conversion");
	return {};
}

template <class R, class F, class In, size_t I, class... T>
typename std::enable_if<I != sizeof...(T), R>::type apply_on(In* in, F const& f)
{
	using ith_t = typename std::tuple_element<I, std::tuple<T...>>::type;
	if (ith_t* v = dynamic_cast<ith_t*>(in)) {
		return f.template call<ith_t>(v);
	}
	return apply_on<R, F, In, I + 1, T...>(in, f);
}
}

template <class R, class... T, class In, class F>
R apply_on(In* p, F&& f)
{
	return __impl::apply_on<R, F, In, 0, T...>(p, std::forward<F>(f));
}

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

namespace
{
struct ith_child {
	template <class T>
	PVCore::PVDataTreeObject* call(T* v) const
	{
		auto children = v->get_children();
		auto it = children.begin();
		std::advance(it, row);
		return *it;
	}

	int row;
};
}

QModelIndex PVHiveDataTreeModel::index(int row, int column, const QModelIndex& parent) const
{
	// Column is always 0 (see columnCount), but asserts it
	assert(column == 0);

	PVCore::PVDataTreeObject* p =
	    (parent.isValid()) ? (PVCore::PVDataTreeObject*)parent.internalPointer() : &_root;

	PVCore::PVDataTreeObject* child =
	    apply_on<PVCore::PVDataTreeObject*, Inendi::PVPlotted, Inendi::PVMapped, Inendi::PVSource>(
	        p, ith_child{row});

	return createIndex(row, column, child);
}

namespace
{
struct get_size {
	template <class T>
	size_t call(T* v) const
	{
		return v->size();
	}
};
}

int PVHiveDataTreeModel::rowCount(const QModelIndex& parent) const
{

	PVCore::PVDataTreeObject* p = (PVCore::PVDataTreeObject*)parent.internalPointer();

	if (not parent.isValid()) {
		return 1;
	} else if (dynamic_cast<Inendi::PVView*>(p)) {
		return 0;
	} else {
		return apply_on<size_t, Inendi::PVPlotted, Inendi::PVMapped, Inendi::PVSource>(p,
		                                                                               get_size{});
	}
}

namespace
{
struct parent_pos {
	template <class T>
	std::tuple<int, PVCore::PVDataTreeObject*> call(T* v) const
	{
		return std::tuple<int, PVCore::PVDataTreeObject*>{_model->pos_from_obj(v->get_parent()),
		                                                  v->get_parent()};
	}

	PVHiveDataTreeModel const* _model;
};
}

QModelIndex PVHiveDataTreeModel::parent(const QModelIndex& index) const
{
	if (!index.isValid()) {
		return {};
	}

	if (index.internalPointer() == &_root) {
		return {};
	}

	PVCore::PVDataTreeObject* id = (PVCore::PVDataTreeObject*)index.internalPointer();

	PVCore::PVDataTreeObject* parent = nullptr;
	int row = 0;

	if (Inendi::PVMapped* v = dynamic_cast<Inendi::PVMapped*>(id)) {
		parent = v->get_parent();
		row = 0;
	} else {
		std::tie(row, parent) =
		    apply_on<std::tuple<int, PVCore::PVDataTreeObject*>, Inendi::PVPlotted, Inendi::PVView>(
		        id, parent_pos{this});
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

namespace
{
struct obj_pos {
	template <class T>
	int call(T* v) const
	{
		auto children = v->get_parent()->get_children();
		return std::distance(children.begin(), std::find(children.begin(), children.end(), v));
	}
};
}

int PVHiveDataTreeModel::pos_from_obj(PVCore::PVDataTreeObject const* id) const
{
	if (dynamic_cast<Inendi::PVSource const*>(id)) {
		return 0;
	} else {
		return apply_on<int, const Inendi::PVPlotted, const Inendi::PVMapped, const Inendi::PVView>(
		    id, obj_pos{});
	}
}
}
