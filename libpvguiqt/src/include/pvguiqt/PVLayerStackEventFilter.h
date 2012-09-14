/**
 * \file PVLayerStackEventFilter.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVLAYERSTACKEVENTFILTER_H
#define PVLAYERSTACKEVENTFILTER_H

#include <QObject>
#include <QEvent>

namespace PVInspector {
class PVMainWindow;
class PVLayerStackView;

/**
 * \class PVLayerStackEventFilter
 */
class PVLayerStackEventFilter : public QObject
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
		PVLayerStackEventFilter(PVMainWindow *mw, PVLayerStackView *parent);

		/**
		 *
		 * @param watched
		 * @param event
		 */
		bool eventFilter(QObject *watched, QEvent *event);
};
}

#endif // PVLAYERSTACKEVENTFILTER_H

