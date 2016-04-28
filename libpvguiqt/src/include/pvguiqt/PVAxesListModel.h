/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_AXESLISTMODEL_H
#define PVGUIQT_AXESLISTMODEL_H

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

class PVAxesListModel : public QAbstractListModel
{
	Q_OBJECT;

  public:
	PVAxesListModel(Inendi::PVView_sp& view_p, QObject* parent = NULL);

  public:
	int rowCount(const QModelIndex& parent) const override;
	int rowCount() const;
	QVariant data(const QModelIndex& index, int role) const override;
	Qt::ItemFlags flags(const QModelIndex& index) const override;

  private slots:
	void about_to_be_deleted_slot(PVHive::PVObserverBase*);
	void refresh_slot(PVHive::PVObserverBase*);

  protected:
	inline Inendi::PVView const& inendi_view() const { return *_view_observer.get_object(); }

  private:
	bool _view_deleted;

	// Observers
	PVHive::PVObserverSignal<Inendi::PVView> _view_observer;
};
}

#endif // __AXESLISTMODEL__H_
