//! \file PVLayerStackEventFilter.h
//! $Id: PVLayerStackEventFilter.h 2496 2011-04-25 14:10:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

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

