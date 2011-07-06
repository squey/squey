//! \file PVLayerStackDelegate.h
//! $Id: PVLayerStackDelegate.h 2496 2011-04-25 14:10:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVLAYERSTACKDELEGATE_H
#define PVLAYERSTACKDELEGATE_H

#include <QStyledItemDelegate>
#include <QWidget>

namespace PVInspector {
class PVMainWindow;
class PVLayerStackView;

/**
 * \class PVLayerStackDelegate
 */
class PVLayerStackDelegate : public QStyledItemDelegate
{
	Q_OBJECT

		PVMainWindow     *main_window;
		PVLayerStackView *layer_stack_view;

	public:
		/**
		 *  Constructor.
		 *
		 *  @param mw
		 *  @param parent
		 */
		PVLayerStackDelegate(PVMainWindow *mw, PVLayerStackView *parent);

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
};
}

#endif // PVLAYERSTACKDELEGATE_H

