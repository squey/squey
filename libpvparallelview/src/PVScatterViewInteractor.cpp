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

#include <pvparallelview/PVScatterViewInteractor.h>
#include <pvparallelview/PVScatterView.h>
#include <pvparallelview/PVScatterViewSelectionRectangle.h>
#include <pvparallelview/PVScatterViewParamsWidget.h>

#include <pvkernel/widgets/PVHelpWidget.h>

/*****************************************************************************
 * PVParallelView::PVScatterViewInteractor::PVScatterViewInteractor
 *****************************************************************************/

PVParallelView::PVScatterViewInteractor::PVScatterViewInteractor(PVWidgets::PVGraphicsView* parent)
    : PVZoomableDrawingAreaInteractor(parent)
{
}

/*****************************************************************************
 * PVParallelView::PVScatterViewInteractor::keyPressEvent
 *****************************************************************************/

bool PVParallelView::PVScatterViewInteractor::keyPressEvent(PVZoomableDrawingArea* zda,
                                                            QKeyEvent* event)
{
	PVScatterView* sv = get_scatter_view(zda);

	if (PVWidgets::PVHelpWidget::is_help_key(event->key())) {
		if (sv->help_widget()->isHidden()) {
			sv->help_widget()->popup(sv->get_viewport(), PVWidgets::PVTextPopupWidget::AlignCenter,
			                         PVWidgets::PVTextPopupWidget::ExpandAll);
			// FIXME : This is a hack to update the help_widget. It should be
			// updated automaticaly as it does with QWebView but it doesn't
			// with QWebEngineView
			sv->raise();
		}
		return false;
	}

	switch (event->key()) {
	case Qt::Key_Escape:
		sv->_sel_rect->clear();
		sv->get_viewport()->update();
		break;
	}

	return false;
}

/*****************************************************************************
 * PVParallelView::PVScatterViewInteractor::resizeEvent
 *****************************************************************************/

bool PVParallelView::PVScatterViewInteractor::resizeEvent(PVZoomableDrawingArea* zda, QResizeEvent*)
{
	PVScatterView* sv = get_scatter_view(zda);

	sv->do_update_all();

	sv->reconfigure_view();

	sv->_sel_rect->set_handles_scale(1. / sv->get_transform().m11(),
	                                 1. / sv->get_transform().m22());

	if (sv->get_viewport()) {
		sv->get_viewport()->update();
	}

	sv->set_params_widget_position();

	return false;
}

/*****************************************************************************
 * PVParallelView::PVScatterViewInteractor::get_scatter_view
 *****************************************************************************/

PVParallelView::PVScatterView*
PVParallelView::PVScatterViewInteractor::get_scatter_view(PVZoomableDrawingArea* zda)
{
	assert(qobject_cast<PVScatterView*>(zda));
	return static_cast<PVScatterView*>(zda);
}
