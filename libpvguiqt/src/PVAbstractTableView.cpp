/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2015
 */

#include <pvguiqt/PVAbstractTableView.h>
#include <pvguiqt/PVAbstractTableModel.h>

#include <QScrollBar>
#include <QHeaderView>
#include <QMouseEvent>

namespace PVGuiQt
{

/******************************************************************************
 *
 * PVAbstractTableView
 *
 *****************************************************************************/

PVAbstractTableView::PVAbstractTableView(QWidget* parent):
	PVTableView(parent)
{
	 connect(verticalScrollBar(), &QScrollBar::valueChanged, this, &PVAbstractTableView::slider_move_to);
	 connect(verticalScrollBar(), &QScrollBar::actionTriggered, this, &PVAbstractTableView::scrollclick);
	 connect(verticalScrollBar(), &QScrollBar::rangeChanged, this,
			 (void (PVAbstractTableView::*)(int, int)) &PVAbstractTableView::new_range);
	 connect(verticalScrollBar(), &QScrollBar::sliderReleased, this, &PVAbstractTableView::clip_slider);

	// Sorting disable as we do it ourself
	setSortingEnabled(false);

	// Text elipsis
	setWordWrap(false);

	setSelectionMode(QAbstractItemView::NoSelection);
}

/******************************************************************************
 *
 * PVAbstractTableView::slider_move_to
 *
 *****************************************************************************/
void PVAbstractTableView::slider_move_to(int value)
{
	if(value == verticalScrollBar()->maximum()) {
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
void PVAbstractTableView::scrollclick(int action) {
	switch(action) {
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
	if(table_model()->is_last_pos() and table_model()->current_page() != 0) {
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
	if(model()) {
		table_model()->update_pages(max - min + 1, verticalScrollBar()->pageStep());
		move_to_page(0);
	}
}

void PVAbstractTableView::new_range()
{
	if(model()) {
		table_model()->update_pages(verticalScrollBar()->maximum() - verticalScrollBar()->minimum() + 1,
				verticalScrollBar()->pageStep());
		move_to_page(0);
	}
}

/******************************************************************************
 *
 * PVAbstractTableView::clip_slider
 *
 *****************************************************************************/
void PVAbstractTableView::clip_slider()
{
	if(verticalScrollBar()->value() == 0) {
		move_to_page(0);
	} else if(verticalScrollBar()->value() == verticalScrollBar()->maximum()) {
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
void PVAbstractTableView::mousePressEvent(QMouseEvent * event)
{
	if(event->button() == Qt::LeftButton) {
		Qt::KeyboardModifiers mod = event->modifiers();
		// Shift and Ctrl continue the selection and don't reset it
		if(not (mod & (Qt::ControlModifier | Qt::ShiftModifier))) {
			table_model()->current_selection().select_none();
		}

		int clc_row = rowAt(event->y());
		if(clc_row < 0) {
			// No row under the mouse.
			return; 
		}

		if(mod & Qt::ShiftModifier) {
			// Shift modifier complete the selection between clicks
			table_model()->end_selection(clc_row);
		} else {
			// Start the selection
			table_model()->start_selection(clc_row);
		}

		// Move below if we click on the half shown row
		int row_pos = rowViewportPosition(clc_row);
		if((row_pos + rowHeight(clc_row) + horizontalHeader()->height()) > (height() + 1)) {
			move_by(1);
		}

		viewport()->update(); // To show the selection
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
			if (not table_model()->current_selection().is_empty()) {
				emit validate_selection();
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
	// Scroll 3 line by wheel step on listing
	move_by(- e->delta() / 8 / 15 * 3);
	e->accept(); // I am the one who handle event
}

/******************************************************************************
 *
 * PVAbstractTableView::mouseReleaseEvent
 *
 *****************************************************************************/
void PVAbstractTableView::mouseReleaseEvent(QMouseEvent * event)
{
	// Mouse release commit the current selection
	table_model()->commit_selection();
	viewport()->update();
	event->accept();
}

/******************************************************************************
 *
 * PVAbstractTableView::mouseMoveEvent
 *
 *****************************************************************************/
void PVAbstractTableView::mouseMoveEvent(QMouseEvent * event)
{
	int pos = event->y();
	// Scroll up while the clicked mouse is above the listing
	while(pos < 0) {
		move_by(-1);
		table_model()->end_selection(rowAt(0));
		pos += rowHeight(rowAt(0));
		if(table_model()->current_page() == 0 and table_model()->pos_in_page() == 0) {
			// We reach the top of the listing, stop scrolling upper
			return;
		}
	}

	int clc_row = rowAt(pos);
	if(clc_row < 0) {
		// We are max up and we keep moving upper
		return;
	}

	// Update selection
	table_model()->end_selection(clc_row);

	// We are in the last partially shown cell, move below
	int row_pos = rowViewportPosition(clc_row);
	if((row_pos + rowHeight(clc_row) + horizontalHeader()->height()) > (height() + 1)) {
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
void PVAbstractTableView::setModel(QAbstractItemModel * model)
{
	PVTableView::setModel(model);
	connect(model, &QAbstractItemModel::layoutChanged, this,
			(void (PVAbstractTableView::*)()) &PVAbstractTableView::new_range);
}

}