/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_AXESCOMBMODEL_H
#define PVGUIQT_AXESCOMBMODEL_H

#include <inendi/PVView.h>

#include <pvkernel/core/PVSharedPointer.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVHive.h>
#include <pvhive/PVObserver.h>
#include <pvhive/PVObserverSignal.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVObserverCallback.h>

#include <QListView>

#define SUBCLASSING_VERSION 0

namespace PVGuiQt
{

class PVAxesCombinationModel;

namespace __impl
{

class set_axis_name_Observer
    : public PVHive::PVFuncObserver<Inendi::PVView,
                                    decltype(&Inendi::PVView::set_axis_name),
                                    &Inendi::PVView::set_axis_name>
{
  public:
	set_axis_name_Observer(PVGuiQt::PVAxesCombinationModel* model) : _model(model) {}

  protected:
	virtual void update(arguments_type const& args) const;

  private:
	PVGuiQt::PVAxesCombinationModel* _model;
};

class remove_column_Observer
    : public PVHive::PVFuncObserver<Inendi::PVView,
                                    decltype(&Inendi::PVView::remove_column),
                                    &Inendi::PVView::remove_column>
{
  public:
	remove_column_Observer(PVGuiQt::PVAxesCombinationModel* model) : _model(model) {}

  protected:
	virtual void about_to_be_updated(arguments_type const& args) const;
	virtual void update(arguments_type const& args) const;

  private:
	PVGuiQt::PVAxesCombinationModel* _model;
};

class axis_append_Observer : public PVHive::PVFuncObserver<Inendi::PVView,
                                                           decltype(&Inendi::PVView::axis_append),
                                                           &Inendi::PVView::axis_append>
{
  public:
	axis_append_Observer(PVGuiQt::PVAxesCombinationModel* model) : _model(model) {}

  protected:
	virtual void update(arguments_type const& args) const;
	virtual void about_to_be_updated(arguments_type const& args) const;

  private:
	PVGuiQt::PVAxesCombinationModel* _model;
};

class move_axis_to_new_position_Observer
    : public PVHive::PVFuncObserver<Inendi::PVView,
                                    decltype(&Inendi::PVView::move_axis_to_new_position),
                                    &Inendi::PVView::move_axis_to_new_position>
{
  public:
	move_axis_to_new_position_Observer(PVGuiQt::PVAxesCombinationModel* model) : _model(model) {}

  protected:
	virtual void update(arguments_type const& args) const;

  private:
	PVGuiQt::PVAxesCombinationModel* _model;
};
}

class PVAxesCombinationModel : public QAbstractListModel
{
	Q_OBJECT;

	friend class PVViewObserver;
	friend class __impl::set_axis_name_Observer;
	friend class __impl::remove_column_Observer;
	friend class __impl::axis_append_Observer;

  public:
	PVAxesCombinationModel(Inendi::PVView_sp& view_p, QObject* parent = NULL);

  public:
	int rowCount(const QModelIndex& parent) const;
	int rowCount() const;
	QVariant data(const QModelIndex& index, int role) const;
	Qt::ItemFlags flags(const QModelIndex& index) const;
	bool setData(const QModelIndex& index, const QVariant& value, int role);
	void beginInsertRow(int row);
	void endInsertRow();
	void beginRemoveRow(int row);
	void endRemoveRow();

  private Q_SLOTS:
	void about_to_be_deleted_slot(PVHive::PVObserverBase*);
	void refresh_slot(PVHive::PVObserverBase*);

  private:
	inline Inendi::PVView const& inendi_view() const { return *_view_observer.get_object(); }

  private:
	PVHive::PVActor<Inendi::PVView> _actor;
	bool _view_deleted;

	// Observers
	__impl::set_axis_name_Observer _set_axis_name_observer;
	PVHive::PVObserverSignal<Inendi::PVView> _view_observer;
	__impl::remove_column_Observer _remove_column_observer;
	__impl::axis_append_Observer _axis_append_observer;
	__impl::move_axis_to_new_position_Observer _move_axis_to_new_position_observer;
	PVHive::PVObserverSignal<Inendi::PVAxesCombination::columns_indexes_t> _obs_axes_comb;
};
}

#endif // __AXESCOMBMODEL__H_
