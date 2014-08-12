/**
 * \file PVZoomedParallelView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H

#include <pvkernel/widgets/PVGraphicsView.h>

namespace PVWidgets
{

class PVHelpWidget;

}

namespace PVParallelView
{

// forward declaration
class PVZoomedParallelScene;
class PVZoomedParallelViewParamsWidget;

/**
 * @class PVZoomedParallelView
 *
 * A derived class of QGraphicsView to use when displaying a zoom view of parallel coordinates
 * representation of events.
 */
class PVZoomedParallelView : public PVWidgets::PVGraphicsView
{
	friend class PVZoomedParallelScene;

public:
	/**
	 * Constructor
	 *
	 * @param parent parent widget
	 */
	PVZoomedParallelView(QWidget *parent = nullptr);

	/**
	 * Destructor
	 */
	~PVZoomedParallelView();

	/**
	 * Overload method when a resize event occurs.
	 *
	 * @param event then resize event
	 */
	void resizeEvent(QResizeEvent *event);

protected:
	PVWidgets::PVHelpWidget* help_widget() { return _help_widget; }
	PVZoomedParallelViewParamsWidget* params_widget() { return _params_widget; }

private:
	PVWidgets::PVHelpWidget          *_help_widget;
	PVZoomedParallelViewParamsWidget *_params_widget;
};

}

#endif // PVPARALLELVIEW_PVZOOMEDPARALLELVIEW_H
