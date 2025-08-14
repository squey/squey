//
// MIT License
//
// © ESI Group, 2015
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

#include <pvguiqt/PVAbstractTableView.h>
#include <pvguiqt/PVAbstractTableModel.h>
#include <pvkernel/core/PVTheme.h>

#include <QScrollBar>
#include <QHeaderView>
#include <QMouseEvent>
#include <QApplication>
#include <QDesktopServices>
#include <QUrl>
#include <QPainter>
#include <QTextCharFormat>
#include <QAbstractTextDocumentLayout>
#include <QTextCursor>
namespace PVGuiQt
{

bool PVHyperlinkDelegate::is_url(const QModelIndex& index) const
{
	QString text = index.data().toString();
	QUrl url(text);
	return url.isValid() and url.scheme().startsWith("http");
}

QString PVHyperlinkDelegate::get_elided_text(const QString& url, const QTextDocument& doc, int max_width) const
{
	QFont font = doc.defaultFont();
	QFontMetrics fm(font);
	return fm.elidedText(url, Qt::ElideRight, max_width);
}

void PVHyperlinkDelegate::format_text_document(QTextDocument& doc, QColor& color, const QString& url, const QRect& rect, bool is_selected /* = false */) const
{
	color = QColor(PVCore::PVTheme::link_colors[(int)PVCore::PVTheme::color_scheme()].name());
	if (is_selected) {
		color = color.darker(300);
	}

	QTextOption text_option;
	text_option.setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
	text_option.setWrapMode(QTextOption::NoWrap);
	doc.setDefaultTextOption(text_option);

	const QString& elided_text = get_elided_text(url, doc, rect.width());
	doc.setHtml(_link.arg(url, elided_text, color.name(), "none"));
}

void PVHyperlinkDelegate::paint(
	QPainter* painter,
	const QStyleOptionViewItem& option,
	const QModelIndex& index) const
{
	if (is_url(index)) {
		QStyleOptionViewItem opt(option);
		initStyleOption(&opt, index);

		const PVAbstractTableModel* model = static_cast<const PVAbstractTableModel*>(index.model());

		painter->save();
		opt.widget->style()->drawPrimitive(QStyle::PE_PanelItemViewItem, &opt, painter, opt.widget);

		QTextDocument doc;
		const QString& url = index.data().toString();
		QColor color;
		format_text_document(doc, color, url, option.rect, model->is_selected(index));
		if (mouse_over_link(index, option.rect)) {
			const QString& elided_text = get_elided_text(url, doc, option.rect.width());
			doc.setHtml(_link.arg(url, elided_text, color.name(), "underline"));
		}


		qreal doc_height = doc.size().height();
		qreal offset_y = option.rect.top() + (option.rect.height() - doc_height) / 2.0;
		painter->translate(option.rect.left(), offset_y);
		doc.setTextWidth(option.rect.width());
		doc.drawContents(painter);
		painter->restore();
	}
	else {
		return QStyledItemDelegate::paint(painter, option, index);
	}
}

bool PVHyperlinkDelegate::mouse_over_link(const QModelIndex& index, const QRect& rect) const
{
	if (is_url(index)) {
		const QString& url = index.data().toString();
		QTextDocument doc;
		QColor color;
		format_text_document(doc, color, url, rect);

		QPointF mouse_local = QPointF(_mouse_pos - rect.topLeft());
		int pos = doc.documentLayout()->hitTest(mouse_local, Qt::ExactHit);

		if (pos != -1) {
			QTextCursor cursor(&doc);
			cursor.setPosition(pos);
			if (not cursor.isNull()) {
				QTextCharFormat format = cursor.charFormat();
				if (format.isAnchor()) {
					const QString& elided_text = get_elided_text(url, doc, rect.width());
					doc.setHtml(_link.arg(url, elided_text, color.name(), "underline"));
					return true;
				}
			}
		}
	}

	return false;
}

void PVHyperlinkDelegate::set_mouse_pos(const QPoint& mouse_pos_in_viewport)
{
	_mouse_pos = mouse_pos_in_viewport;
}

bool PVAbstractTableView::viewportEvent(QEvent* event)
{
    switch (event->type()) {
		case QEvent::MouseMove: {
			auto* e = static_cast<QMouseEvent*>(event);
			if (_hyperlink_delegate) {
                _hyperlink_delegate->set_mouse_pos(e->pos());
                viewport()->update();
            }
			QModelIndex index = indexAt(e->pos());
			QString text = index.data().toString();
			QUrl url(text);

			if (_hyperlink_delegate and _hyperlink_delegate->mouse_over_link(index, visualRect(index))) {
				setCursor(Qt::PointingHandCursor);
			} else {
				unsetCursor();
			}
			break;
		}
		case QEvent::Leave: {
			if (_hyperlink_delegate) {
                _hyperlink_delegate->set_mouse_pos(QPoint(-1, -1));
                viewport()->update();
            }
			unsetCursor();
			break;
		}
		case QEvent::MouseButtonPress: {
			auto* e = static_cast<QMouseEvent*>(event);
			QModelIndex index = indexAt(e->pos());
			if (not index.isValid()) {
				break;
			}

			QString text = index.data().toString();
			QUrl url(text);
			if (_hyperlink_delegate and _hyperlink_delegate->mouse_over_link(index, visualRect(index))) {
				QDesktopServices::openUrl(url);
				return true;
			}
			break;
		}
		default: {
			break;
		}
    }

    return PVTableView::viewportEvent(event);
}

/******************************************************************************
 *
 * PVAbstractTableView
 *
 *****************************************************************************/

PVAbstractTableView::PVAbstractTableView(QWidget* parent) : PVTableView(parent)
{
	// Handle hyperlink rendering
	_hyperlink_delegate = new PVHyperlinkDelegate(this);
	setMouseTracking(true);
	viewport()->setMouseTracking(true);

	connect(verticalScrollBar(), &QScrollBar::valueChanged, this,
	        &PVAbstractTableView::slider_move_to);
	connect(verticalScrollBar(), &QScrollBar::actionTriggered, this,
	        &PVAbstractTableView::scrollclick);
	connect(verticalScrollBar(), &QScrollBar::rangeChanged, this,
	        (void (PVAbstractTableView::*)(int, int)) & PVAbstractTableView::new_range);
	connect(verticalScrollBar(), &QScrollBar::sliderReleased, this,
	        &PVAbstractTableView::clip_slider);

	setSortingEnabled(true);

	// Text elipsis
	setWordWrap(false);

	setSelectionMode(QAbstractItemView::NoSelection);

	// enabling QSS for headers
	horizontalHeader()->setObjectName("horizontalHeader_of_PVAbstractTableView");
	verticalHeader()->setObjectName("verticalHeader_of_PVAbstractTableView");

	// Show contextual menu on right click in the table (set menuPolicy to emit
	// signals)
	connect(this, &QWidget::customContextMenuRequested, this,
	        &PVAbstractTableView::show_rclick_menu);
	setContextMenuPolicy(Qt::CustomContextMenu);
}

/******************************************************************************
 *
 * PVAbstractTableView::set_hyperlink_delegate_max_index
 *
 *****************************************************************************/
void PVAbstractTableView::set_hyperlink_delegate_max_index(int max_index)
{
	_hyperlink_delegate_max_index = max_index;
}

/******************************************************************************
 *
 * PVAbstractTableView::show_rclick_menu
 *
 *****************************************************************************/
void PVAbstractTableView::show_rclick_menu(QPoint const& p)
{
	table_model()->commit_selection();
	show_ctxt_menu(p);
}

/******************************************************************************
 *
 * PVAbstractTableView::slider_move_to
 *
 *****************************************************************************/
void PVAbstractTableView::slider_move_to(int value)
{
	if (value == verticalScrollBar()->maximum()) {
		// Move to the end of the listing
		move_to_end();
	} else {
		// Move to the top of the page
		move_to_page(value);
	}
	viewport()->update();
	verticalHeader()->viewport()->update();
}

/******************************************************************************
 *
 * PVAbstractTableView::scrollclick
 *
 *****************************************************************************/
void PVAbstractTableView::scrollclick(int action)
{
	switch (action) {
	case QAbstractSlider::SliderSingleStepAdd:
		move_by(1);
		break;
	case QAbstractSlider::SliderSingleStepSub:
		move_by(-1);
		break;
	case QAbstractSlider::SliderPageStepAdd:
		move_by(verticalScrollBar()->pageStep());
		break;
	case QAbstractSlider::SliderPageStepSub:
		move_by(-verticalScrollBar()->pageStep());
		break;
	case QAbstractSlider::SliderToMinimum:
		move_to_page(0);
		break;
	case QAbstractSlider::SliderToMaximum:
		move_to_end();
		break;
	}
}

/******************************************************************************
 *
 * PVAbstractTableView::move_by
 *
 *****************************************************************************/
void PVAbstractTableView::move_by(int row)
{
	table_model()->move_by(row, verticalScrollBar()->pageStep());
	update_on_move();
}

/******************************************************************************
 *
 * PVAbstractTableView::move_to_nraw
 *
 *****************************************************************************/
void PVAbstractTableView::move_to_nraw(PVRow row)
{
	table_model()->move_to_nraw(row, verticalScrollBar()->pageStep());
	update_on_move();
}

/******************************************************************************
 *
 * PVAbstractTableView::move_to_row
 *
 *****************************************************************************/
void PVAbstractTableView::move_to_row(PVRow row)
{
	table_model()->move_to_row(row, verticalScrollBar()->pageStep());
	update_on_move();
}

/******************************************************************************
 *
 * PVAbstractTableView::move_to_page
 *
 *****************************************************************************/
void PVAbstractTableView::move_to_page(int page)
{
	table_model()->move_to_page(page);
	update_on_move();
}

/******************************************************************************
 *
 * PVAbstractTableView::move_to_end
 *
 *****************************************************************************/
void PVAbstractTableView::move_to_end()
{
	table_model()->move_to_end(verticalScrollBar()->pageStep());
	update_on_move();
}

/******************************************************************************
 *
 * PVAbstractTableView::update_on_move
 *
 *****************************************************************************/
void PVAbstractTableView::update_on_move()
{
	// Save and restore pos_in_range as moving cursor call slider_move_to which
	// set pos_in_page_to 0.
	size_t pos_in_page = table_model()->pos_in_page();
	// Check if there is a scrollbar, otherwise current_page can be 0 but
	// setValue(1) is invalid.
	if (table_model()->is_last_pos() and table_model()->current_page() != 0) {
		// Last tick is only use when we reach the end
		verticalScrollBar()->setValue(table_model()->current_page() + 1);
	} else {
		verticalScrollBar()->setValue(table_model()->current_page());
	}
	table_model()->pos_in_page() = pos_in_page;
	verticalHeader()->viewport()->update();
	viewport()->update();
}

/******************************************************************************
 *
 * PVAbstractTableView::new_range
 *
 *****************************************************************************/
void PVAbstractTableView::new_range(int min, int max)
{
	if (model() and table_model()->size() > 0) {
		// min == max means we have only the current page so it contains every lines
		// without
		// scroll. The page size must be big enought to get them all.
		// Save previously selected nraw line
		size_t nraw_pos = table_model()->current_nraw_line();
		size_t step = verticalScrollBar()->pageStep();
		table_model()->update_pages(max - min + 1, step);
		auto const& shown_lines = table_model()->shown_lines();
		// Keep this line on top if it is still present
		if (std::find(shown_lines.begin(), shown_lines.end(), nraw_pos) != shown_lines.end()) {
			move_to_nraw(nraw_pos);
		} else {
			// Otherwise, go to the top
			move_to_page(0);
		}
	}
}

void PVAbstractTableView::new_range()
{
	new_range(verticalScrollBar()->minimum(), verticalScrollBar()->maximum());
}

/******************************************************************************
 *
 * PVAbstractTableView::clip_slider
 *
 *****************************************************************************/
void PVAbstractTableView::clip_slider()
{
	if (verticalScrollBar()->value() == 0) {
		move_to_page(0);
	} else if (verticalScrollBar()->value() == verticalScrollBar()->maximum()) {
		move_to_end();
	}
}

/******************************************************************************
 *
 * PVAbstractTableView::table_model
 *
 *****************************************************************************/
PVAbstractTableModel* PVAbstractTableView::table_model()
{
	return static_cast<PVAbstractTableModel*>(model());
}

/******************************************************************************
 *
 * PVAbstractTableView::mousePressEvent
 *
 *****************************************************************************/
void PVAbstractTableView::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		Qt::KeyboardModifiers mod = event->modifiers();
		int clc_row = rowAt(event->position().y());

		if (clc_row < 0) {
			// No valid row under the mouse
			if (mod == Qt::NoModifier) {
				// Reset the whole selection only if there is no used modifier
				table_model()->reset_selection();
			}
			return;
		}

		if (mod == Qt::ShiftModifier) {
			// Change the range selection end position in set selection mode
			table_model()->set_selection_mode(PVAbstractTableModel::SET);
			table_model()->end_selection(clc_row);
		} else if (mod == Qt::ControlModifier) {
			// Start the range selection by getting the start row state an applying it
			// to other rows
			table_model()->commit_selection();
			table_model()->set_selection_mode(PVAbstractTableModel::TOGGLE_AND_USE);
			table_model()->start_selection(clc_row);
		} else if (mod == (Qt::ShiftModifier | Qt::ControlModifier)) {
			// Start the range selection in invert selection mode
			table_model()->commit_selection();
			table_model()->set_selection_mode(PVAbstractTableModel::NEGATE);
			table_model()->start_selection(clc_row);
		} else if (mod == Qt::NoModifier) {
			// Reset the selection and start the range selection in set selection mode
			table_model()->reset_selection();
			table_model()->set_selection_mode(PVAbstractTableModel::SET);
			table_model()->start_selection(clc_row);
		}

		// Move below if we click on the half shown row
		int row_pos = rowViewportPosition(clc_row);
		if ((row_pos + rowHeight(clc_row) + horizontalHeader()->height()) > (height() + 1)) {
			move_by(1);
		}

		table_model()->commit_selection();

	} else if (event->button() == Qt::RightButton) {
		QModelIndex index = indexAt(event->pos());

		if ((index.isValid()) && (not table_model()->is_selected(index))) {
			int clc_row = rowAt(event->position().y());

			table_model()->reset_selection();
			table_model()->start_selection(clc_row);
		}
	}
	else {
		PVTableView::mousePressEvent(event);
	}
}

/******************************************************************************
 *
 * PVAbstractTableView::keyPressEvent
 *
 *****************************************************************************/
void PVAbstractTableView::keyPressEvent(QKeyEvent* event)
{
	switch (event->key()) {
	case Qt::Key_Return:
	case Qt::Key_Enter:
		if (table_model()->has_selection()) {
			table_model()->commit_selection();
			Q_EMIT validate_selection();
		}
		break;

	// Bind document displacement key
	case Qt::Key_PageUp:
		scrollclick(QAbstractSlider::SliderPageStepSub);
		break;
	case Qt::Key_PageDown:
		scrollclick(QAbstractSlider::SliderPageStepAdd);
		break;
	case Qt::Key_Up:
		scrollclick(QAbstractSlider::SliderSingleStepSub);
		break;
	case Qt::Key_Down:
		scrollclick(QAbstractSlider::SliderSingleStepAdd);
		break;
	case Qt::Key_Home:
		scrollclick(QAbstractSlider::SliderToMinimum);
		break;
	case Qt::Key_End:
		scrollclick(QAbstractSlider::SliderToMaximum);
		break;
	case Qt::Key_Right:
		horizontalScrollBar()->triggerAction(QAbstractSlider::SliderSingleStepAdd);
		break;
	case Qt::Key_Left:
		horizontalScrollBar()->triggerAction(QAbstractSlider::SliderSingleStepSub);
		break;
	case Qt::Key_A:
		if ((event->modifiers() == Qt::NoModifier) || (event->modifiers() & Qt::ControlModifier)) {
			table_model()->reset_selection();
			table_model()->current_selection().select_all();
			viewport()->update();
		}
		break;
	case Qt::Key_I:
		if ((event->modifiers() == Qt::NoModifier) || (event->modifiers() & Qt::ControlModifier)) {
			table_model()->commit_selection();
			table_model()->clear_selection();
			table_model()->current_selection().select_inverse();
			viewport()->update();
		}
		break;
	default:
		PVTableView::keyPressEvent(event);
	}
}

/******************************************************************************
 *
 * PVAbstractTableView::wheelEvent
 *
 *****************************************************************************/
void PVAbstractTableView::wheelEvent(QWheelEvent* e)
{
	// delta is wheel movement in degree. QtWheelEvent doc give this formule
	// to convert it to "wheel step"
	// http://doc.qt.io/qt-5/qwheelevent.html
	double complete_scroll_angle = 120.0;
	double delta_y = e->angleDelta().y() / 8.0 * 15.0;

	// anti-sticky : reset accumulator when changing scrolling direction
	if (delta_y * _scroll_accumulator_y < 0) {
		_scroll_accumulator_y = 0;
	}
	_scroll_accumulator_y += delta_y;

	if (std::abs(_scroll_accumulator_y) >= complete_scroll_angle) {
		move_by(_scroll_accumulator_y > 0 ? -1 : 1);
		_scroll_accumulator_y += _scroll_accumulator_y > 0 ? -complete_scroll_angle : complete_scroll_angle;
	}
	e->accept(); // I am the one who handle event
}

/******************************************************************************
 *
 * PVAbstractTableView::mouseReleaseEvent
 *
 *****************************************************************************/
void PVAbstractTableView::mouseReleaseEvent(QMouseEvent* event)
{
	if (table_model()->current_selection().is_empty()) {
		return;
	}
	viewport()->update();
	event->accept();
	table_model()->commit_selection();
	Q_EMIT selection_commited();
}

/******************************************************************************
 *
 * PVAbstractTableView::mouseMoveEvent
 *
 *****************************************************************************/
void PVAbstractTableView::mouseMoveEvent(QMouseEvent* event)
{
	if (event->buttons() == Qt::NoButton) {
        return;
    }

	int pos = event->position().y();
	// Scroll up while the clicked mouse is above the listing
	while (pos < 0) {
		move_by(-1);
		table_model()->end_selection(rowAt(0));
		pos += rowHeight(rowAt(0));
		if (table_model()->current_page() == 0 and table_model()->pos_in_page() == 0) {
			// We reach the top of the listing, stop scrolling upper
			return;
		}
	}

	int clc_row = rowAt(pos);
	if (clc_row < 0) {
		// We are max up and we keep moving upper
		return;
	}

	// Update selection
	table_model()->end_selection(clc_row);

	// We are in the last partially shown cell, move below
	int row_pos = rowViewportPosition(clc_row);
	if ((row_pos + rowHeight(clc_row) + horizontalHeader()->height()) > (height() + 1)) {
		move_by(1);
	}

	viewport()->update(); // Show selection modification
	event->accept();
}

/******************************************************************************
 *
 * PVAbstractTableView::setModel
 *
 *****************************************************************************/
void PVAbstractTableView::setModel(QAbstractItemModel* model)
{
	PVTableView::setModel(model);

	// Setup model hyperlink delegate
	auto setup_hyperlink_delegate = [this,model]() {
		if (_hyperlink_delegate) {
			int column_count = _hyperlink_delegate_max_index == 0 ? model->columnCount() : _hyperlink_delegate_max_index;
			for (int col = 0; col < column_count; ++col) {
				setItemDelegateForColumn(col, _hyperlink_delegate);
			}
		}
	};
	setup_hyperlink_delegate();
	connect(table_model(), &QAbstractItemModel::modelReset,
	this, [setup_hyperlink_delegate]() {
		setup_hyperlink_delegate();
	});

	connect(model, &QAbstractItemModel::layoutChanged, this,
	        (void (PVAbstractTableView::*)()) & PVAbstractTableView::new_range);
	Q_EMIT table_model()->layoutChanged();
}
} // namespace PVGuiQt
