/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
} // namespace PVWidgets

namespace PVParallelView
{

class PVRenderingJob;
class PVFullParallelScene;
class PVFullParallelViewParamsWidget;

class PVFullParallelView : public QGraphicsView
{
	Q_OBJECT
	friend class PVFullParallelScene;

  public:
	explicit PVFullParallelView(QWidget* parent = nullptr);
	~PVFullParallelView() override;

  public:
	void set_total_events_number(uint32_t total_events_number)
	{
		_total_events_number = total_events_number;
	}
	void set_selected_events_number(uint32_t selected_events_number)
	{
		_selected_events_number = selected_events_number;
	}
	void set_axes_number(uint32_t axes_number) { _axes_number = axes_number; }

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
	PVFullParallelViewParamsWidget* _params_widget;

	uint32_t _total_events_number = 0;
	uint32_t _selected_events_number = 0;
	uint32_t _axes_number = 0;
	bool _first_resize;
};
} // namespace PVParallelView

#endif // __PVFULLPARALLELVIEW_H__
