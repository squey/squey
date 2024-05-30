//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/widgets/PVHelpWidget.h>

#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZoomedParallelViewParamsWidget.h>

#include <QGuiApplication>
#include <QScrollBar>
#include <QPainter>

PVParallelView::PVZoomedParallelView::PVZoomedParallelView(
    Squey::PVAxesCombination const& axes_comb, QWidget* parent)
    : PVWidgets::PVGraphicsView(parent)
{
	setMinimumHeight(300);

	install_default_scene_interactor();

	_help_widget = new PVWidgets::PVHelpWidget(this);
	_help_widget->hide();

	_help_widget->initTextFromFile("zoomed parallel view's help");
	_help_widget->addTextFromFile(":help-selection");
	_help_widget->addTextFromFile(":help-layers");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-lines");
	_help_widget->addTextFromFile(":help-application");

	_help_widget->newTable();
	_help_widget->addTextFromFile(":help-mouse-zoomed-parallel-view");
	_help_widget->finalizeText();

	_params_widget = new PVZoomedParallelViewParamsWidget(axes_comb, this);
	_params_widget->adjustSize();

	_mouse_buttons_default_legend = PVWidgets::PVMouseButtonsLegend("Select", "Pan view", "Zoom");
	_mouse_buttons_current_legend = _mouse_buttons_default_legend;
}

PVParallelView::PVZoomedParallelView::~PVZoomedParallelView()
{
	if (get_scene()) {
		delete get_scene();
	}
}

void PVParallelView::PVZoomedParallelView::resizeEvent(QResizeEvent* event)
{
	PVWidgets::PVGraphicsView::resizeEvent(event);

	if (auto zps = static_cast<PVParallelView::PVZoomedParallelScene*>(get_scene())) {
		_params_widget->move({0, 0});
		_params_widget->adjustSize();
		_params_widget->raise();

		bool need_recomputation = event->oldSize().height() != event->size().height();
		zps->resize_display(need_recomputation);
	}
}

void PVParallelView::PVZoomedParallelView::update_window_title(Squey::PVView& view, PVCombCol combcol)
{
	setWindowTitle(QString("%1 (%2)").arg(QObject::tr("Zoomed"), view.get_axis_name(combcol)));
}

void PVParallelView::PVZoomedParallelView::enterEvent(QEnterEvent* /*event*/)
{
	if (QGuiApplication::keyboardModifiers() == Qt::ControlModifier) {
		_mouse_buttons_current_legend.set_scrollwheel_legend("Pan view (1px)");
	}
	else if (QGuiApplication::keyboardModifiers() == Qt::ShiftModifier) {
		_mouse_buttons_current_legend.set_scrollwheel_legend("Pan view (step)");
	}
	Q_EMIT set_status_bar_mouse_legend(_mouse_buttons_current_legend);
	setFocus(Qt::MouseFocusReason);
}

void PVParallelView::PVZoomedParallelView::leaveEvent(QEvent*)
{
	Q_EMIT clear_status_bar_mouse_legend();
	_mouse_buttons_current_legend = _mouse_buttons_default_legend;
	clearFocus();
}

void PVParallelView::PVZoomedParallelView::keyPressEvent(QKeyEvent* event)
{
	if (event->modifiers() == Qt::ControlModifier) {
		_mouse_buttons_current_legend.set_scrollwheel_legend("Pan view (1px)");
		Q_EMIT set_status_bar_mouse_legend(_mouse_buttons_current_legend);
	}
	else if (event->modifiers() == Qt::ShiftModifier) {
		_mouse_buttons_current_legend.set_scrollwheel_legend("Pan view (step)");
		Q_EMIT set_status_bar_mouse_legend(_mouse_buttons_current_legend);
	}

	PVWidgets::PVGraphicsView::keyPressEvent(event);
}

void PVParallelView::PVZoomedParallelView::keyReleaseEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_Control) {
		_mouse_buttons_current_legend.set_scrollwheel_legend("Zoom");
		Q_EMIT set_status_bar_mouse_legend(_mouse_buttons_current_legend);
	}
	else if (event->key() == Qt::Key_Shift) {
		_mouse_buttons_current_legend.set_scrollwheel_legend("Zoom");
		Q_EMIT set_status_bar_mouse_legend(_mouse_buttons_current_legend);
	}

	PVWidgets::PVGraphicsView::keyReleaseEvent(event);
}