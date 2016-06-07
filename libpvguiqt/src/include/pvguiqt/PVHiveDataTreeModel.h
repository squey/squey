/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef DATATREEMODEL_H
#define DATATREEMODEL_H

#include <pvhive/PVHive.h>
#include <pvhive/PVObserverSignal.h>
#include <inendi/PVSource.h>

#include <QAbstractItemModel>

namespace PVGuiQt
{

class PVHiveDataTreeModel : public QAbstractItemModel
{
	Q_OBJECT

	using datatree_obs_t = PVHive::PVObserverSignal<PVCore::PVDataTreeObject>;

  public:
	PVHiveDataTreeModel(Inendi::PVSource& root, QObject* parent = nullptr);
	QModelIndex index(int row, int column, const QModelIndex& parent) const override;

  protected:
	int rowCount(const QModelIndex& index) const override;
	int columnCount(const QModelIndex&) const override { return 1; }

	QVariant data(const QModelIndex& index, int role) const override;

	Qt::ItemFlags flags(const QModelIndex& /*index*/) const override
	{
		return Qt::ItemIsSelectable | Qt::ItemIsEnabled | Qt::ItemIsUserCheckable;
	}

	QModelIndex parent(const QModelIndex& index) const override;

  private slots:
	void hive_refresh(PVHive::PVObserverBase* o);

	void about_to_be_deleted(PVHive::PVObserverBase*)
	{
		beginResetModel();
		endResetModel();
	}

  private:
	template <class T>
	void register_obs(T* o)
	{
		this->_obs.emplace_back(static_cast<QObject*>(this));
		datatree_obs_t* obs = &_obs.back();
		auto datatree_o = o->shared_from_this();
		PVHive::get().register_observer(datatree_o, *obs);
		obs->connect_refresh(this, SLOT(hive_refresh(PVHive::PVObserverBase*)));
		obs->connect_about_to_be_deleted(this, SLOT(about_to_be_deleted(PVHive::PVObserverBase*)));
	}

	void register_all_observers()
	{
		for (auto& mapped : _root.get_children()) {
			if (not is_object_observed(mapped.get())) {
				register_obs(mapped.get());
				beginResetModel();
				endResetModel();
			}
			for (auto& plotted : mapped->get_children()) {
				if (not is_object_observed(mapped.get())) {
					register_obs(plotted.get());
					beginResetModel();
					endResetModel();
				}
				for (auto& view : plotted->get_children()) {
					if (not is_object_observed(mapped.get())) {
						register_obs(view.get());
						beginResetModel();
						endResetModel();
					}
				}
			}
		}
	}

	bool is_object_observed(PVCore::PVDataTreeObject* o) const
	{
		return std::find_if(_obs.begin(), _obs.end(), [o](datatree_obs_t const& obs) {
			       return obs.get_object() == o;
			   }) != _obs.end();
	}

	int pos_from_obj(PVCore::PVDataTreeObject const* o) const;

  private:
	Inendi::PVSource& _root;
	std::list<datatree_obs_t> _obs;
	PVHive::PVObserver_p<PVCore::PVDataTreeObject> _root_recursive_observer;
};
}

#endif
