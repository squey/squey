/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#include <QTreeView>

#ifndef __RUSH_PVERFTREEVIEW_H__
#define __RUSH_PVERFTREEVIEW_H__

#include "PVERFTreeModel.h"

#include <pvkernel/core/serialize_numbers.h>

#include <QTreeView>
#include <QEvent>

namespace PVRush
{

class PVERFTreeView : public QTreeView
{
	Q_OBJECT;

  public:
	PVERFTreeView(PVRush::PVERFTreeModel* model, QWidget* parent = nullptr);

  public:
	bool select(const rapidjson::Document& json);

  private:
	void set_item_state(QModelIndex index, Qt::CheckState state);
	void set_children_state(QModelIndex index, Qt::CheckState state);
	void set_parents_state(QModelIndex index);

  private:
	void currentChanged(const QModelIndex& current, const QModelIndex& previous) override
	{
		QTreeView::currentChanged(current, previous);
		Q_EMIT current_changed(current, previous);
	}

  Q_SIGNALS:
	void current_changed(const QModelIndex&, const QModelIndex&);
	void model_changed();

  private:
	PVRush::PVERFTreeModel* _model;
	bool _changing_selection = false;
	bool _changing_check_state = false;
};

} // namespace PVRush

#endif