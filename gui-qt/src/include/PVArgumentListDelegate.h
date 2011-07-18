//! \file PVArgumentListDelegate.h
//! $Id: PVArgumentListDelegate.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVARGUMENTLISTDELEGATE_H
#define PVARGUMENTLISTDELEGATE_H

#include <QtCore>

#include <QPainter>
#include <QStyledItemDelegate>
#include <QTableView>

#include <pvcore/general.h>
#include <pvfilter/PVArgument.h>
#include <picviz/PVView.h>

namespace PVInspector {

class PVArgumentListDelegate : public QStyledItemDelegate {
public:
	PVArgumentListDelegate(Picviz::PVView& view, QTableView* parent = 0);
public:
	QWidget* createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;
	bool editorEvent(QEvent * event, QAbstractItemModel * model, const QStyleOptionViewItem & option, const QModelIndex & index);	
	void setEditorData(QWidget* editor, const QModelIndex& index) const;
	void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
	void updateEditorGeometry(QWidget* editor, const QStyleOptionViewItem &option, const QModelIndex &index) const;
	QSize sizeHint(const QStyleOptionViewItem & option, const QModelIndex & index) const;
	QSize getSize() const;
	/* int sizeHintForColumn(int column) const; */
	void paint(QPainter* painter, const QStyleOptionViewItem &option, const QModelIndex &index) const;
protected:
	QTableView* _parent;
};

}

#endif	/* PVARGUMENTLISTDELEGATE_H */

