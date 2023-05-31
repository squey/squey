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

#include <cmath>

#include <QEvent>
#include <QMetaType>
#include <QVariant>

#include <squey/PVSelection.h>
#include <squey/PVStateMachine.h>
#include <squey/PVView.h>

#include <pvguiqt/PVLayerStackDelegate.h>

#include <QPainter>

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackDelegate::PVLayerStackDelegate
 *
 *****************************************************************************/
PVGuiQt::PVLayerStackDelegate::PVLayerStackDelegate(Squey::PVView const& view, QObject* parent)
    : QStyledItemDelegate(parent), _view(view)
{
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackDelegate::editorEvent
 *
 *****************************************************************************/
bool PVGuiQt::PVLayerStackDelegate::editorEvent(QEvent* event,
                                                QAbstractItemModel* model,
                                                const QStyleOptionViewItem& /*option*/,
                                                const QModelIndex& index)
{
	switch (index.column()) {
	case 0:
		if (event->type() == QEvent::MouseButtonPress) {
			model->setData(index, QVariant());
			/* We start by reprocessing only the layer_stack */
			return true;
		}
		break;
	}
	return false;
}
