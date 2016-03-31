/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <cmath>

#include <QEvent>
#include <QMetaType>
#include <QVariant>

#include <inendi/PVSelection.h>
#include <inendi/PVStateMachine.h>
#include <inendi/PVView.h>

#include <pvguiqt/PVLayerStackDelegate.h>

#include <QPainter>

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackDelegate::PVLayerStackDelegate
 *
 *****************************************************************************/
PVGuiQt::PVLayerStackDelegate::PVLayerStackDelegate(Inendi::PVView const& view, QObject* parent):
	QStyledItemDelegate(parent),
	_view(view)
{
}

/******************************************************************************
 *
 * PVGuiQt::PVLayerStackDelegate::editorEvent
 *
 *****************************************************************************/
bool PVGuiQt::PVLayerStackDelegate::editorEvent(QEvent *event, QAbstractItemModel* model, const QStyleOptionViewItem &/*option*/, const QModelIndex &index)
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
