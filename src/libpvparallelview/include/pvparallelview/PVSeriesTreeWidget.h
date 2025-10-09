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

#ifndef __PVSERIESTREEWIDGET_H__
#define __PVSERIESTREEWIDGET_H__

#include <QAbstractItemModel>
#include <QModelIndex>
#include <QVariant>
#include <QVector>
#include <QTreeView>
#include <QPainter>
#include <QStyledItemDelegate>
#include <QSortFilterProxyModel>
#include <QKeyEvent>
#include <QDebug>

#include <assert.h>

#include <squey/PVView.h>
#include <squey/PVRangeSubSampler.h>
#include <pvkernel/rush/PVAxisFormat.h>
#include <pvkernel/rush/PVNraw.h>

class PVSeriesTreeItem
{
  public:
	explicit PVSeriesTreeItem(const QString& text = QString(),
	                          const QColor& color = QColor(),
	                          PVCol data = PVCol(),
	                          PVSeriesTreeItem* parent_item = nullptr)
	    : _parent_item(parent_item)
	    , _item_text(text)
	    , _item_color(color)
	    , _item_data(QVariant::fromValue(data))
	{
	}

	~PVSeriesTreeItem() { qDeleteAll(_child_items); }

	void append_child(PVSeriesTreeItem* child) { _child_items.append(child); };
	int child_count() const { return _child_items.count(); }
	int column_count() const { return 1; }
	PVSeriesTreeItem* parent_item() { return _parent_item; }

	PVSeriesTreeItem* child(int row)
	{
		if (row < 0 or row >= _child_items.size()) {
			return nullptr;
		}
		return _child_items.at(row);
	}

	QVariant data(int role) const
	{
		if (role == Qt::DisplayRole or role == Qt::ToolTipRole) {
			return _item_text;
		} else if (role == Qt::BackgroundRole) {
			return QBrush(_item_color.value<QColor>());
		} else if (role == Qt::UserRole) {
			return _item_data;
		} else {
			// assert(false);
			return QVariant();
		}
	}

	int row() const
	{
		if (_parent_item) {
			return _parent_item->_child_items.indexOf(const_cast<PVSeriesTreeItem*>(this));
		}

		return 0;
	}

  private:
	PVSeriesTreeItem* _parent_item;
	QVector<PVSeriesTreeItem*> _child_items;

	QVariant _item_text;
	QVariant _item_color;
	QVariant _item_data;
};

class PVSeriesTreeModel : public QAbstractItemModel
{
	Q_OBJECT

  public:
	explicit PVSeriesTreeModel(
	    Squey::PVView* view,
	    const Squey::PVRangeSubSampler& sampler /*const QString &data, QObject *parent = nullptr*/)
	    : QAbstractItemModel(/*parent*/ nullptr)
	{
		_root_item = new PVSeriesTreeItem();
		setup_model_data(view, sampler);
	}
	~PVSeriesTreeModel() { delete _root_item; }

	QVariant data(const QModelIndex& index, int role) const override
	{
		if (not index.isValid()) {
			return QVariant();
		}

		PVSeriesTreeItem* item = static_cast<PVSeriesTreeItem*>(index.internalPointer());

		return item->data(role);
	}

	Qt::ItemFlags flags(const QModelIndex& index) const override
	{
		if (not index.isValid()) {
			return Qt::NoItemFlags;
		}

		return QAbstractItemModel::flags(index);
	}

	QVariant
	headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override
	{
		if (orientation == Qt::Horizontal && role == Qt::DisplayRole) {
			return _root_item->data(section);
		}
		return QVariant();
	}

	QModelIndex index(int row, int column, const QModelIndex& parent = QModelIndex()) const override
	{
		if (not hasIndex(row, column, parent)) {
			return QModelIndex();
		}

		PVSeriesTreeItem* parent_item;

		if (not parent.isValid()) {
			parent_item = _root_item;
		} else {
			parent_item = static_cast<PVSeriesTreeItem*>(parent.internalPointer());
		}

		PVSeriesTreeItem* child_item = parent_item->child(row);
		if (child_item) {
			return createIndex(row, column, child_item);
		}

		return QModelIndex();
	}

	QModelIndex parent(const QModelIndex& index) const override
	{
		if (!index.isValid()) {
			return QModelIndex();
		}

		PVSeriesTreeItem* child_item = static_cast<PVSeriesTreeItem*>(index.internalPointer());
		PVSeriesTreeItem* parent_item = child_item->parent_item();

		if (parent_item == _root_item) {
			return QModelIndex();
		}

		return createIndex(parent_item->row(), 0, parent_item); // "0" ?
	}

	int rowCount(const QModelIndex& parent = QModelIndex()) const override
	{
		PVSeriesTreeItem* parent_item;
		if (parent.column() > 0) {
			return 0;
		}

		if (not parent.isValid()) {
			parent_item = _root_item;
		} else {
			parent_item = static_cast<PVSeriesTreeItem*>(parent.internalPointer());
		}

		return parent_item->child_count();
	}

	int columnCount(const QModelIndex& parent = QModelIndex()) const override
	{
		if (parent.isValid()) {
			return static_cast<PVSeriesTreeItem*>(parent.internalPointer())->column_count();
		}

		return _root_item->column_count();
	}

  private:
	void setup_model_data(Squey::PVView* _view, const Squey::PVRangeSubSampler& sampler)
	{
		const Squey::PVAxesCombination& axes_comb = _view->get_axes_combination();
		PVCol column_count = _view->get_rushnraw_parent().column_count();
		for (PVCol col(0); col < column_count; col++) {
			const PVRush::PVAxisFormat& axis = axes_comb.get_axis(col);
			if (axis.get_type().startsWith("number_") or axis.get_type().startsWith("duration")) {
				QColor color(rand() % 156 + 100, rand() % 156 + 100, rand() % 156 + 100);
				PVSeriesTreeItem* item =
				    new PVSeriesTreeItem(axis.get_name(), color, col, _root_item);
				_root_item->append_child(item);
				for (size_t i = 0; i < sampler.group_count() and sampler.group_count() > 1; i++) {
					PVSeriesTreeItem* subitem =
					    new PVSeriesTreeItem(sampler.group_name(i).c_str(), color,
					                         PVCol(sampler.group_count() * col + i), item);
					item->append_child(subitem);
				}
			}
		}
	}

  private:
	PVSeriesTreeItem* _root_item;
};

class PVSeriesTreeFilterProxyModel : public QSortFilterProxyModel
{
	Q_OBJECT

  public:
	void set_selection(const QSet<int>& selected)
	{
		_selected = selected;
		beginFilterChange();
		endFilterChange();
	}

	void clear_selection() { _selected.clear(); }

  protected:
	bool filterAcceptsRow(int source_row, const QModelIndex& source_parent) const override
	{
		const QModelIndex& index = sourceModel()->index(source_row, 0, source_parent);
		PVCol index_id = index.data(Qt::UserRole).value<PVCol>();
		QString text = index.data(Qt::DisplayRole).value<QString>();
		return _selected.contains(index_id) or _selected.isEmpty();
	}

  private:
	QSet<int> _selected;
};

class PVSeriesTreeStyleDelegate : public QStyledItemDelegate
{
  public:
	explicit PVSeriesTreeStyleDelegate(QWidget* parent = nullptr) : QStyledItemDelegate(parent) {}
	void paint(QPainter* painter,
	           const QStyleOptionViewItem& option,
	           const QModelIndex& index) const override
	{
		QColor color = index.model()->data(index, Qt::BackgroundRole).value<QBrush>().color();
		if ((option.state & QStyle::State_Selected)) {
			painter->fillRect(option.rect, color);
			painter->setPen(Qt::black);
		} else {
			painter->setPen(color);
		}
		painter->drawText(option.rect, Qt::AlignLeft,
		                  index.model()->data(index, Qt::DisplayRole).toString());
	}
};

class PVSeriesTreeView : public QTreeView
{
	Q_OBJECT

  public:
	explicit PVSeriesTreeView(bool filtered = false) : QTreeView(nullptr), _filtered(filtered)
	{
		setSelectionMode(QAbstractItemView::MultiSelection);
		setItemDelegate(new PVSeriesTreeStyleDelegate());
	}

  public:
	using QTreeView::selectedIndexes;

	void keyPressEvent(QKeyEvent* event)
	{
		if ((event->key() == Qt::Key_A and (event->modifiers() == Qt::ControlModifier)) or
		    (event->modifiers() == Qt::ShiftModifier)) {
			event->ignore();
			return;
		}
		QTreeView::keyPressEvent(event);
	}

  public:
	void setSelectionModel(QItemSelectionModel* model)
	{
		QTreeView::setSelectionModel(model);
		connect(selectionModel(), &QItemSelectionModel::selectionChanged, this,
		        &PVSeriesTreeView::propagate_selection_to_children);
	}

  Q_SIGNALS:
	void selection_changed();

  private:
	void propagate_selection_to_children(const QItemSelection& selected_items,
	                                     const QItemSelection& unselected_items)
	{
		if (_filtered) {
			return;
		}

		struct filtered_guard {
			PVSeriesTreeView* p;
			explicit filtered_guard(decltype(p) parent) : p(parent) { p->_filtered = true; }
			~filtered_guard() { p->_filtered = false; }
		} fguard{this};

		QItemSelection new_total_sel(selected_items);
		QItemSelection new_total_unsel(unselected_items);

		bool unsel = false;
		bool sel = false;

		// select children
		for (const QModelIndex& index : selected_items.indexes()) {
			if (not index.parent().isValid()) { // top level item
				sel = true;
				new_total_sel.merge(
				    QItemSelection(model()->index(0, 0, index), model()->index(model()->rowCount(index) - 1, 0, index)),
				    QItemSelectionModel::Select);
			}
		}

		if (sel) {
			selectionModel()->select(new_total_sel, QItemSelectionModel::SelectCurrent);
		}

		// unselect children
		QSet<QModelIndex> parents;
		for (const QModelIndex& index : unselected_items.indexes()) {
			if (not index.parent().isValid()) { // top level item
				unsel = true;
				new_total_unsel.merge(QItemSelection(index, index), QItemSelectionModel::Select);
				new_total_unsel.merge(
				    QItemSelection(model()->index(0, 0, index), model()->index(model()->rowCount(index) - 1, 0, index)),
				    QItemSelectionModel::Select);
			} else {
				parents.insert(index.parent());
			}
		}

		// Unselect parents too if all children are unselected
		for (const QModelIndex& parent : parents) {
			bool all_children_unselected = true;
			for (PVCol row(0); row < model()->rowCount(parent) and all_children_unselected; row++) {
				const QModelIndex& child = model()->index(row, 0, parent);
				all_children_unselected &= not selectionModel()->isSelected(child);
			}
			if (all_children_unselected) {
				new_total_unsel.merge(QItemSelection(parent, parent), QItemSelectionModel::Select);
				unsel = true;
			}
		}

		if (unsel) {
			selectionModel()->select(new_total_unsel, QItemSelectionModel::Deselect);
		}

		Q_EMIT selection_changed();
	}

  private:
	bool _filtered;
};

#endif // __PVSERIESTREEWIDGET_H__
