/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
	void select(const rapidjson::Document& json);

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
