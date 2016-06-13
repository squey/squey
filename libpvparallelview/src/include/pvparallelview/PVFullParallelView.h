/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVFULLPARALLELVIEW_H__
#define __PVFULLPARALLELVIEW_H__

#include <QGraphicsView>
#include <QScrollBar>
#include <QFuture>
#include <QFontMetrics>
#include <QLocale>

#include <iostream>

#include <pvparallelview/common.h>

#include <inendi/PVView.h>

class QEvent;

namespace PVWidgets
{

class PVHelpWidget;
}

namespace PVParallelView
{

class PVRenderingJob;
class PVFullParallelScene;

class PVFullParallelView : public QGraphicsView
{
	Q_OBJECT
	friend class PVFullParallelScene;

  public:
	PVFullParallelView(QWidget* parent = NULL);
	~PVFullParallelView();

  public:
	void set_total_events_number(uint32_t total_events_number)
	{
		_total_events_number = total_events_number;
	}
	void set_selected_events_number(uint32_t selected_events_number)
	{
		_selected_events_number = selected_events_number;
	}

  protected:
	void resizeEvent(QResizeEvent* event) override;
	void enterEvent(QEvent* event) override;
	void leaveEvent(QEvent* event) override;

	void drawForeground(QPainter* painter, const QRectF& rect) override;

	/**
	 * simulate a mouse move event
	 */
	void fake_mouse_move();

	PVWidgets::PVHelpWidget* help_widget() { return _help_widget; }

  Q_SIGNALS:
	void new_zoomed_parallel_view(Inendi::PVView* view, int axis_index);

  private:
	PVWidgets::PVHelpWidget* _help_widget;

	uint32_t _total_events_number = 0;
	uint32_t _selected_events_number = 0;
	bool _first_resize;
};
}

#endif // __PVFULLPARALLELVIEW_H__
