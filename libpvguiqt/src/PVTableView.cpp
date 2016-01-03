/**
 * @file
 *
 * 
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include <QEvent>
#include <QHelpEvent>
#include <QModelIndex>
#include <QRect>
#include <QToolTip>

#include <pvguiqt/PVTableView.h>

/******************************************************************************
 *
 * PVGuiQt::PVTableView::viewportEvent
 *
 *****************************************************************************/
bool PVGuiQt::PVTableView::viewportEvent(QEvent *event) {
	if (event->type() == QEvent::ToolTip) {
		// Check if the text is elided. If it is, keep going, otherwise, hide
		// the current ToolTip (from another cell maybe) and intercept the
		// event.
		QHelpEvent *helpEvent = static_cast<QHelpEvent*>(event);
		// We don't need to care about row reindexing as row are fixed height
		// and column give same width for every row
		QModelIndex index = indexAt(helpEvent->pos());
		if (index.isValid()) {
			QSize sizeHint = itemDelegate(index)->sizeHint(viewOptions(), index);
			QRect rItem(0, 0, sizeHint.width(), sizeHint.height());
			QRect rVisual = visualRect(index);
			if (rItem.width() <= rVisual.width()) {
				QToolTip::hideText();
				return true;
			}
		}
	}
	return QTableView::viewportEvent(event);
}

/******************************************************************************
 *
 * PVGuiQt::PVTableView::resizeEvent
 *
 *****************************************************************************/
void PVGuiQt::PVTableView::resizeEvent(QResizeEvent *event) {
	QTableView::resizeEvent(event);
	emit resize();
}
