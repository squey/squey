//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "ColumnsPanel.h"

#include "Style.h"

#include <libpvpcap/ws.h>

#include <pvkernel/widgets/PVImportWidget.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <QDragEnterEvent>
#include <QDropEvent>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QListWidget>
#include <QMimeData>
#include <QPainter>
#include <QScrollBar>
#include <QStyledItemDelegate>
#include <QToolButton>
#include <QVBoxLayout>

#include <algorithm>
#include <functional>

namespace PVPcapsicum
{

namespace
{

enum role { field_role = Qt::UserRole, type_role, name_role, details_role, flags_role };
enum flag { two_passes = 1, unknown = 2, empty = 4 };

QRect close_area(const QRect& item)
{
	const int side = 20;
	return {item.right() - side - 6, item.center().y() - side / 2, side, side};
}

//! The field of a column, what it is, and an example of its values.
class ColumnDelegate : public QStyledItemDelegate
{
  public:
	using QStyledItemDelegate::QStyledItemDelegate;

	QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex&) const override
	{
		return {option.rect.width(), option.fontMetrics.height() * 2 + 12};
	}

	void paint(QPainter* painter,
	           const QStyleOptionViewItem& option,
	           const QModelIndex& index) const override
	{
		QStyleOptionViewItem background = option;
		initStyleOption(&background, index);
		background.text.clear();
		option.widget->style()->drawControl(QStyle::CE_ItemViewItem, &background, painter,
		                                    option.widget);

		painter->save();
		const bool selected = option.state & QStyle::State_Selected;
		const bool hovered = option.state & QStyle::State_MouseOver;
		const QColor text =
		    option.palette.color(selected ? QPalette::HighlightedText : QPalette::Text);
		const QColor secondary =
		    selected ? option.palette.color(QPalette::HighlightedText) : style::secondary();
		const QRect area = option.rect.adjusted(8, 5, hovered ? -34 : -8, -5);
		const int line = option.fontMetrics.height();
		const QRect first(area.left(), area.top(), area.width(), line);
		const QRect second(area.left() + 28, area.top() + line + 2, area.width() - 28, line);

		QFont small = option.font;
		small.setPointSizeF(small.pointSizeF() * 0.9);
		painter->setFont(small);
		painter->setPen(secondary);
		painter->drawText(QRect(first.left(), first.top(), 24, line),
		                  Qt::AlignRight | Qt::AlignVCenter, QString::number(index.row() + 1));

		const int flags = index.data(flags_role).toInt();
		painter->setFont(option.font);
		painter->setPen(text);
		int x = first.left() + 28;
		const QString field = option.fontMetrics.elidedText(
		    index.data(field_role).toString(), Qt::ElideMiddle, first.right() - x - 70);
		painter->drawText(QRect(x, first.top(), first.right() - x, line),
		                  Qt::AlignLeft | Qt::AlignVCenter, field);
		x += option.fontMetrics.horizontalAdvance(field) + 8;
		x += style::paint_tag(*painter, x, first, index.data(type_role).toString(), secondary) + 4;
		if (flags & unknown) {
			x += style::paint_tag(*painter, x, first, QObject::tr("unknown"), style::error()) + 4;
		}
		if (flags & empty) {
			x += style::paint_tag(*painter, x, first, QObject::tr("empty"), style::warning()) + 4;
		}
		if (flags & two_passes) {
			style::paint_tag(*painter, x, first, QObject::tr("2 passes"), style::warning());
		}

		const QString details = index.data(details_role).toString();
		painter->setFont(small);
		painter->setPen(secondary);
		painter->drawText(second, Qt::AlignLeft | Qt::AlignVCenter,
		                  QFontMetrics(small).elidedText(details, Qt::ElideRight, second.width()));

		if (hovered) {
			const QRect close = close_area(option.rect);
			PVModdedIcon("window-close").paint(painter, close.adjusted(3, 3, -3, -3));
		}
		painter->restore();
	}
};

//! Takes fields dropped from the tree of fields, and the keys moving and removing columns.
class ColumnList : public QListWidget
{
  public:
	struct actions {
		std::function<void(int)> remove_row;
		std::function<void()> remove_selected;
		std::function<void(int)> move_selected;
		std::function<void(const QStringList&, int)> dropped;
	};

	explicit ColumnList(actions a) : _actions(std::move(a))
	{
		setMouseTracking(true);
		viewport()->setAttribute(Qt::WA_Hover);
	}

  protected:
	void keyPressEvent(QKeyEvent* event) override
	{
		if (event->matches(QKeySequence::Delete) or event->key() == Qt::Key_Backspace) {
			_actions.remove_selected();
			return;
		}
		if (event->modifiers() & Qt::AltModifier and
		    (event->key() == Qt::Key_Up or event->key() == Qt::Key_Down)) {
			_actions.move_selected(event->key() == Qt::Key_Up ? -1 : 1);
			return;
		}
		QListWidget::keyPressEvent(event);
	}

	void mousePressEvent(QMouseEvent* event) override
	{
		QListWidgetItem* under = itemAt(event->position().toPoint());
		if (under != nullptr and
		    close_area(visualItemRect(under)).contains(event->position().toPoint())) {
			_actions.remove_row(row(under));
			return;
		}
		QListWidget::mousePressEvent(event);
	}

	void dragEnterEvent(QDragEnterEvent* event) override
	{
		if (event->mimeData()->hasFormat(FieldTreeModel::MIME_TYPE)) {
			event->acceptProposedAction();
			return;
		}
		QListWidget::dragEnterEvent(event);
	}

	void dragMoveEvent(QDragMoveEvent* event) override
	{
		if (event->mimeData()->hasFormat(FieldTreeModel::MIME_TYPE)) {
			_drop_row = drop_row(event->position().toPoint());
			viewport()->update();
			event->acceptProposedAction();
			return;
		}
		QListWidget::dragMoveEvent(event);
	}

	void dragLeaveEvent(QDragLeaveEvent* event) override
	{
		_drop_row = -1;
		viewport()->update();
		QListWidget::dragLeaveEvent(event);
	}

	void dropEvent(QDropEvent* event) override
	{
		if (event->mimeData()->hasFormat(FieldTreeModel::MIME_TYPE)) {
			const QStringList fields =
			    QString::fromUtf8(event->mimeData()->data(FieldTreeModel::MIME_TYPE))
			        .split('\n', Qt::SkipEmptyParts);
			const int row = drop_row(event->position().toPoint());
			_drop_row = -1;
			viewport()->update();
			event->acceptProposedAction();
			_actions.dropped(fields, row);
			return;
		}
		QListWidget::dropEvent(event);
	}

	void paintEvent(QPaintEvent* event) override
	{
		QListWidget::paintEvent(event);
		QPainter painter(viewport());
		if (count() == 0) {
			painter.setPen(style::secondary());
			painter.drawText(viewport()->rect().adjusted(16, 16, -16, -16),
			                 Qt::AlignCenter | Qt::TextWordWrap,
			                 tr("No column yet: check fields on the left, or drag them here."));
		}
		if (_drop_row >= 0) {
			const int y = _drop_row < count() ? visualItemRect(item(_drop_row)).top()
			              : count() == 0      ? 2
			                                  : visualItemRect(item(count() - 1)).bottom();
			painter.setPen(QPen(style::accent(), 2));
			painter.drawLine(4, y, viewport()->width() - 4, y);
		}
	}

  private:
	int drop_row(const QPoint& at) const
	{
		const QListWidgetItem* under = itemAt(at);
		if (under == nullptr) {
			return count();
		}
		const QRect area = visualItemRect(under);
		return row(under) + (at.y() > area.center().y() ? 1 : 0);
	}

	actions _actions;
	int _drop_row = -1;
};

} // namespace

ColumnsPanel::ColumnsPanel(QWidget* parent) : QWidget(parent)
{
	setAttribute(Qt::WA_StyledBackground);
	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(6);

	auto* header = new QHBoxLayout;
	header->addWidget(PVWidgets::PVImportWidget::section_title(tr("Columns")));
	_count = new QLabel;
	style::color_label(_count, style::secondary(), true);
	header->addWidget(_count);
	header->addStretch();
	const auto tool = [&](const char* icon, const QString& tip, const char* name) {
		auto* button = new QToolButton;
		button->setObjectName(name);
		button->setIcon(PVModdedIcon(icon));
		button->setToolTip(tip);
		button->setAutoRaise(true);
		button->setEnabled(false);
		header->addWidget(button);
		return button;
	};
	_up = tool("arrow-up", tr("Move the selected columns up (Alt+Up)"), "column_up_button");
	_down =
	    tool("arrow-down", tr("Move the selected columns down (Alt+Down)"), "column_down_button");
	_remove =
	    tool("trash-xmark", tr("Remove the selected columns (Delete)"), "remove_columns_button");
	layout->addLayout(header);

	_list = new ColumnList(
	    {[this](int row) { remove_row(row); }, [this]() { remove_selected(); },
	     [this](int by) { move_selected(by); },
	     [this](const QStringList& fields, int row) { Q_EMIT dropped(fields, row); }});
	_list->setObjectName("columns_list");
	_list->setItemDelegate(new ColumnDelegate(_list));
	_list->setSelectionMode(QAbstractItemView::ExtendedSelection);
	_list->setDragDropMode(QAbstractItemView::InternalMove);
	_list->setDefaultDropAction(Qt::MoveAction);
	layout->addWidget(_list, 1);

	connect(_up, &QToolButton::clicked, this, [this]() { move_selected(-1); });
	connect(_down, &QToolButton::clicked, this, [this]() { move_selected(1); });
	connect(_remove, &QToolButton::clicked, this, &ColumnsPanel::remove_selected);
	connect(_list, &QListWidget::itemSelectionChanged, this, &ColumnsPanel::update_header);
	connect(_list->model(), &QAbstractItemModel::rowsMoved, this, &ColumnsPanel::emit_edited);
	connect(_list, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem* item) {
		Q_EMIT reveal(item->data(field_role).toString().toStdString());
	});
	update_header();
}

void ColumnsPanel::set_catalog(std::shared_ptr<const pvpcap::field_catalog> catalog)
{
	_catalog = std::move(catalog);
	refresh();
}

void ColumnsPanel::set_samples(const sampled_fields& samples)
{
	_samples = samples;
	refresh();
}

void ColumnsPanel::set_columns(const std::vector<pvpcap::profile_field>& columns)
{
	if (columns == _columns) {
		return;
	}
	_columns = columns;
	refresh();
}

void ColumnsPanel::set_filled(const std::map<std::string, double>& filled)
{
	if (filled == _filled) {
		return;
	}
	_filled = filled;
	refresh();
}

void ColumnsPanel::refresh()
{
	// Updated in place while the same columns are shown, which keeps where the list is.
	bool same = _list->count() == int(_columns.size());
	for (int i = 0; same and i < _list->count(); i++) {
		same = _list->item(i)->data(field_role).toString().toStdString() == _columns[i].name;
	}
	const QSignalBlocker blocker(_list);
	if (not same) {
		QStringList selected;
		for (const QListWidgetItem* item : _list->selectedItems()) {
			selected << item->data(field_role).toString();
		}
		const int scrolled = _list->verticalScrollBar()->value();
		_list->clear();
		for (const pvpcap::profile_field& column : _columns) {
			auto* item = new QListWidgetItem(_list);
			item->setData(field_role, QString::fromStdString(column.name));
			item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable | Qt::ItemIsDragEnabled);
			item->setSelected(selected.contains(QString::fromStdString(column.name)));
		}
		_list->verticalScrollBar()->setValue(scrolled);
	}

	for (int i = 0; i < _list->count(); i++) {
		QListWidgetItem* item = _list->item(i);
		const pvpcap::profile_field& column = _columns[i];
		const QString name = QString::fromStdString(column.name);
		item->setData(type_role, style::type_text(column.type));
		int flags = 0;
		QStringList details;
		QString tip = QString("<b>%1</b>").arg(name.toHtmlEscaped());
		if (_catalog) {
			const int64_t field = _catalog->find_field(column.name);
			if (field >= 0) {
				const auto& known = _catalog->fields()[field];
				details << style::to_qstring(known.name);
				tip += " — " + style::to_qstring(known.name).toHtmlEscaped();
				if (not known.description.empty()) {
					tip += "<br>" + style::to_qstring(known.description).toHtmlEscaped();
				}
			} else {
				flags |= unknown;
				tip += "<br>" + tr("tshark does not list this field, which then comes out empty, "
				                   "or keeps tshark from reading the captures: it may come from "
				                   "an earlier version of Wireshark.");
				// Wireshark 3.0 renamed SSL into TLS, and kept the old names, always empty.
				const std::string renamed = "tls." + column.name.substr(4);
				if (column.name.starts_with("ssl.") and _catalog->find_field(renamed) >= 0) {
					tip +=
					    "<br>" + tr("tshark now names it %1.").arg(QString::fromStdString(renamed));
				}
			}
		}
		tip +=
		    "<br>" + tr("%1, imported as %2")
		                 .arg(QString::fromStdString(column.type), style::type_text(column.type));
		if (const auto seen = _samples.fields.find(column.name); seen != _samples.fields.end()) {
			details << tr("e.g. %1").arg(QString::fromStdString(seen->second.example));
		}
		if (const auto filled = _filled.find(column.name); filled != _filled.end()) {
			if (filled->second == 0) {
				flags |= empty;
				tip += "<br><i>" + tr("None of the rows of the preview has a value.") + "</i>";
			} else if (filled->second < 0.995) {
				details
				    << tr("in %1% of the rows").arg(std::max(1, int(filled->second * 100 + 0.5)));
			}
		}
		if (pvpcap::ws_needs_two_passes(column.name, column.type)) {
			flags |= two_passes;
			tip += "<br><i>" +
			       tr("tshark only writes it right when it reads the captures twice, which takes "
			          "about a quarter longer.") +
			       "</i>";
		}
		item->setData(details_role, details.join(" · "));
		item->setData(flags_role, flags);
		item->setToolTip(tip);
	}
	update_header();
}

void ColumnsPanel::emit_edited()
{
	std::vector<pvpcap::profile_field> columns;
	for (int i = 0; i < _list->count(); i++) {
		const std::string name = _list->item(i)->data(field_role).toString().toStdString();
		const auto it = std::find_if(_columns.begin(), _columns.end(),
		                             [&](const auto& column) { return column.name == name; });
		if (it != _columns.end()) {
			columns.push_back(*it);
		}
	}
	_columns = columns;
	_list->viewport()->update();
	update_header();
	Q_EMIT edited(columns);
}

void ColumnsPanel::move_selected(int by)
{
	std::vector<int> rows;
	for (const QListWidgetItem* item : _list->selectedItems()) {
		rows.push_back(_list->row(item));
	}
	if (rows.empty()) {
		return;
	}
	std::sort(rows.begin(), rows.end());
	if ((by < 0 and rows.front() == 0) or (by > 0 and rows.back() == _list->count() - 1)) {
		return;
	}
	std::vector<pvpcap::profile_field> columns = _columns;
	if (by < 0) {
		for (const int row : rows) {
			std::swap(columns[row], columns[row - 1]);
		}
	} else {
		for (auto it = rows.rbegin(); it != rows.rend(); ++it) {
			std::swap(columns[*it], columns[*it + 1]);
		}
	}
	_columns = columns;
	refresh();
	Q_EMIT edited(columns);
}

void ColumnsPanel::remove_selected()
{
	std::vector<pvpcap::profile_field> columns;
	for (int i = 0; i < _list->count(); i++) {
		if (not _list->item(i)->isSelected()) {
			columns.push_back(_columns[i]);
		}
	}
	if (columns.size() == _columns.size()) {
		return;
	}
	const int current = _list->currentRow();
	_columns = columns;
	refresh();
	if (_list->count() != 0) {
		_list->setCurrentRow(std::min(current, _list->count() - 1));
	}
	Q_EMIT edited(columns);
}

void ColumnsPanel::remove_row(int row)
{
	std::vector<pvpcap::profile_field> columns = _columns;
	columns.erase(columns.begin() + row);
	_columns = columns;
	refresh();
	Q_EMIT edited(columns);
}

void ColumnsPanel::update_header()
{
	_count->setText(_columns.empty() ? QString() : QString::number(_columns.size()));
	const bool any = not _list->selectedItems().isEmpty();
	_up->setEnabled(any);
	_down->setEnabled(any);
	_remove->setEnabled(any);
}

} // namespace PVPcapsicum
