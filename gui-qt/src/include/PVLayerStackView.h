//! \file
//! $Id: PVLayerStackView.h 2496 2011-04-25 14:10:00Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVLAYERSTACKVIEW_H
#define PVLAYERSTACKVIEW_H


#include <QTableView>
#include <QEvent>

#include <PVLayerStackDelegate.h>
#include <PVLayerStackEventFilter.h>
#include <PVLayerStackModel.h>

namespace PVInspector {
class PVMainWindow;
class PVLayerStackWidget;

/**
 *  \class PVLayerStackView
 */
class PVLayerStackView : public QTableView
{
	Q_OBJECT

		PVMainWindow *main_window;

		PVLayerStackDelegate *layer_stack_delegate;
		PVLayerStackEventFilter *layer_stack_event_filter;

	public:
		int mouse_hover_layer_index; // FIXME, private
		int last_mouse_hover_layer_index; // FIXME, private
		PVLayerStackView(PVMainWindow *mw, PVLayerStackModel *model, PVLayerStackWidget *parent);

		void leaveEvent(QEvent *event);
//		void set_model(PVLayerStackModel *pv_listing_model);
};
}

#endif // PVLAYERSTACKVIEW_H
