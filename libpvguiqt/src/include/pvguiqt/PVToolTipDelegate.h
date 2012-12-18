/**
 * \file PVToolTipDelegate.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVGUIQT_PVTOOLTIPDELEGATE_H__
#define __PVGUIQT_PVTOOLTIPDELEGATE_H__

class QHelpEvent;
class QAbstractItemView;
class QStyleOptionViewItem;
class QModelIndex;
#include <QStyledItemDelegate>
#include <QObject>

namespace PVGuiQt
{

class PVToolTipDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    PVToolTipDelegate(QWidget* parent) : QStyledItemDelegate(parent) {}

public slots:
    bool helpEvent(QHelpEvent* e, QAbstractItemView* view, const QStyleOptionViewItem& option, const QModelIndex& index);
};

}


#endif // __PVGUIQT_PVTOOLTIPDELEGATE_H__
