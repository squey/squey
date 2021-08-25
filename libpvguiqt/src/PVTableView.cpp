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
bool PVGuiQt::PVTableView::viewportEvent(QEvent* event)
{
	if (event->type() == QEvent::ToolTip) {
		// Check if the text is elided. If it is, keep going, otherwise, hide
		// the current ToolTip (from another cell maybe) and intercept the
		// event.
		QHelpEvent* helpEvent = static_cast<QHelpEvent*>(event);
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
void PVGuiQt::PVTableView::resizeEvent(QResizeEvent* event)
{
	QTableView::resizeEvent(event);
	Q_EMIT resize();
}
