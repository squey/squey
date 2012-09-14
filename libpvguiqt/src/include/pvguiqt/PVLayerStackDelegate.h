/**
 * \file PVLayerStackDelegate.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVLAYERSTACKDELEGATE_H
#define PVLAYERSTACKDELEGATE_H

#include <picviz/PVView_types.h>

#include <QStyledItemDelegate>

namespace PVGuiQt {

/**
 * \class PVLayerStackDelegate
 */
class PVLayerStackDelegate : public QStyledItemDelegate
{
	Q_OBJECT

public:
	/**
	 *  Constructor.
	 *
	 *  @param mw
	 *  @param parent
	 */
	PVLayerStackDelegate(Picviz::PVView const& view, QObject* parent = NULL);

	/**
	 *  @param parent
	 *  @param option
	 *  @param index
	 *
	 *  @return
	 */
	QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;

	/**
	 *  @param event
	 *  @param model
	 *  @param option
	 *  @param index
	 *
	 *  @return
	 */
	bool editorEvent (QEvent *event, QAbstractItemModel *model, const QStyleOptionViewItem &option, const QModelIndex &index);

	/**
	 *  @param painter
	 *  @param option
	 *  @param index
	 *
	 */
	void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index ) const;

	/**
	 *  @param editor
	 *  @param index
	 *
	 */
	void setEditorData(QWidget *editor, const QModelIndex &index) const;

	/**
	 * @param editor
	 * @param model
	 * @param index
	 *
	 */
	void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;

	/**
	 *
	 * @param option
	 * @param index
	 *
	 * @return
	 */
	QSize sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index ) const;

private:
	Picviz::PVView const& lib_view() const { return _view; }

private:
	Picviz::PVView const& _view;
};

}

#endif // PVLAYERSTACKDELEGATE_H

