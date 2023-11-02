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

#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVFullParallelScene.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVFullParallelViewParamsWidget.h>

#include <QPaintEvent>
#include <QApplication>
#include <QScrollBar>

/******************************************************************************
 *
 * PVParallelView::PVFullParallelView::PVFullParallelView
 *
 *****************************************************************************/
PVParallelView::PVFullParallelView::PVFullParallelView(QWidget* parent)
    : QGraphicsView(parent), _first_resize(true)
{
	viewport()->setCursor(Qt::CrossCursor);
	setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
	setMinimumHeight(300);
	setFrameShape(QFrame::NoFrame);

	((QScrollBar*)verticalScrollBar())->setObjectName("verticalScrollBar_of_PVListingView");
	((QScrollBar*)horizontalScrollBar())->setObjectName("horizontalScrollBar_of_PVListingView");

	_help_widget = new PVWidgets::PVHelpWidget(this);
	_help_widget->hide();

	_help_widget->initTextFromFile("full parallel view's help", ":help-style");
	_help_widget->addTextFromFile(":help-selection");
	_help_widget->addTextFromFile(":help-layers");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-lines");
	_help_widget->addTextFromFile(":help-application");

	_help_widget->newTable();
	_help_widget->addTextFromFile(":help-mouse-view");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-sel-rect-simple");

	_help_widget->newTable();
	_help_widget->addTextFromFile(":help-mouse-full-parallel-view");
	_help_widget->newColumn();
	_help_widget->addTextFromFile(":help-shortcuts-full-parallel-view");
	_help_widget->finalizeText();

	_params_widget = new PVFullParallelViewParamsWidget(this);

	_params_widget->setAutoFillBackground(true);
	_params_widget->adjustSize();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelView::~PVFullParallelView
 *
 *****************************************************************************/
PVParallelView::PVFullParallelView::~PVFullParallelView()
{
	PVLOG_DEBUG("In PVFullParallelView destructor\n");
	if (scene()) {
		delete scene();
	}
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelView::paintEvent
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelView::drawForeground(QPainter* painter, const QRectF& rect)
{
	// Get back in viewport's coordinates system
	painter->save();
	painter->resetTransform();

	QRectF rect_view = mapFromScene(rect).boundingRect();

	const QString sel_text = QString("%L1").arg(_selected_events_number);
	const QString sep_text = QString(" /");
	const QString total_text = QString(" %L1 Rows").arg(_total_events_number);
	const QString percent_prefix_text(" (");
	const QString percent_suffix_text(" %)");
	const QString percent_text = QString("%1").arg(
	    (uint32_t)(100.0 * (double)_selected_events_number / (double)_total_events_number), 3);
	const QString axes_number_text = QString(" %L1 Axes").arg(_axes_number);

	/* to have a fixed sized frame, the selection count size is deduced from the total count
	 * value (without the extra space)
	 */
	const QString max_sel_text = QString("%L1").arg(_total_events_number);

	const QColor sel_col(QRgb(0x71b4eb));
	const QColor percent_col = sel_col;

	QFont f(painter->font());
	f.setWeight(QFont::Bold);
	painter->setFont(f);

	QFontMetrics fm(painter->font());

	const QSize sel_size = fm.size(Qt::TextSingleLine, sel_text);
	const QSize max_sel_size = fm.size(Qt::TextSingleLine, max_sel_text);
	const QSize sep_size = fm.size(Qt::TextSingleLine, sep_text);
	const QSize total_size = fm.size(Qt::TextSingleLine, total_text);
	const QSize percent_prefix_size = fm.size(Qt::TextSingleLine, percent_prefix_text);
	const QSize percent_suffix_size = fm.size(Qt::TextSingleLine, percent_suffix_text);
	const QSize percent_size = fm.size(Qt::TextSingleLine, percent_text);
	const QSize axes_number_text_size = fm.size(Qt::TextSingleLine, axes_number_text);
	const QSize percent_spacing_size = fm.size(Qt::TextSingleLine, "000");

	const int text_width = max_sel_size.width() + sep_size.width() + total_size.width() +
	                       percent_prefix_size.width() + percent_suffix_size.width() +
	                       percent_spacing_size.width() + axes_number_text_size.width();
	const int text_height =
	    std::max(std::max(std::max(max_sel_size.height(), sep_size.height()),
	                      std::max(total_size.height(), percent_spacing_size.height())),
	             std::max(percent_prefix_size.height(), percent_suffix_size.height()));

	const int frame_width = text_width + frame_margins.left() + frame_margins.right();
	const QRect frame(width() - frame_width - frame_offsets.left(), frame_offsets.top(),
	                  frame_width, text_height + frame_margins.top() + frame_margins.bottom());

	const QSize text_size(text_width, text_height);

	/* the "stats" frame
	 */
	painter->setPen(Qt::NoPen);

	painter->drawRect(frame);

	/* The "stats" strings are drawn only if necessary
	 */
	QPoint text_pos(frame.left() + frame_margins.left() + max_sel_size.width() - sel_size.width(),
	                frame.top() + frame_margins.top() + fm.ascent());

	if (QRectF(text_pos, text_size).intersects(rect_view)) {
		painter->setPen(sel_col);
		painter->drawText(text_pos, sel_text);
		text_pos.rx() += sel_size.width();

		painter->setPen(frame_text_color);
		painter->drawText(text_pos, sep_text);
		text_pos.rx() += sep_size.width();

		painter->drawText(text_pos, total_text);
		text_pos.rx() += total_size.width();

		painter->setPen(percent_col);

		painter->drawText(text_pos, percent_prefix_text);
		text_pos.rx() += percent_prefix_size.width();

		// a right alignment
		text_pos.rx() += percent_spacing_size.width() - percent_size.width();

		painter->drawText(text_pos, percent_text);
		text_pos.rx() += percent_size.width();

		painter->drawText(text_pos, percent_suffix_text);
		text_pos.rx() += percent_suffix_size.width();

		painter->setPen(frame_text_color);
		painter->drawText(text_pos, axes_number_text);
	}

#ifdef SQUEY_DEVELOPER_MODE
	if (common::show_bboxes()) {
		const QPolygonF scene_rect = mapFromScene(scene()->sceneRect());
		painter->setPen(QPen(QColor(0xFF, 0, 0), 0));
		painter->setBrush(QColor(0xFF, 0, 0, 40));
		painter->drawPolygon(scene_rect);

		painter->setPen(QPen(QColor(0xf6, 0xf2, 0x40), 0));
		const QPolygonF items_rect = mapFromScene(scene()->itemsBoundingRect());
		painter->drawPolygon(items_rect);

		painter->setPen(QPen(QColor(0x00, 0x00, 0xFF), 0));
		QList<QGraphicsItem*> pixmaps = scene()->items();
		for (QGraphicsItem* p : pixmaps) {
			if (dynamic_cast<QGraphicsPixmapItem*>(p)) {
				painter->drawPolygon(mapFromScene(p->sceneBoundingRect()));
			}
		}
	}
#endif

	painter->restore();
}

/******************************************************************************
 *
 * PVParallelView::PVFullParallelView::resizeEvent
 *
 *****************************************************************************/
void PVParallelView::PVFullParallelView::resizeEvent(QResizeEvent* event)
{
	QGraphicsView::resizeEvent(event);

	auto* fps = (PVParallelView::PVFullParallelScene*)scene();
	if (fps != nullptr) {
		fps->update_viewport();
		if (_first_resize) {
			_first_resize = false;
			fps->reset_zones_layout_to_default();
		} else {
			fps->update_scene(false);
		}
		fps->scale_all_zones_images();
		fps->update_all_with_timer();

		/* to force the view to be always at the top. Otherwise,
		 * resizing the window to a smaller size automatically translates
		 * the view in a wrong way.
		 */
		verticalScrollBar()->setValue(verticalScrollBar()->minimum());
	}
}

/*****************************************************************************
 * PVParallelView::PVFullParallelView::enterEvent
 *****************************************************************************/

void PVParallelView::PVFullParallelView::enterEvent(QEnterEvent*)
{
	
	setFocus(Qt::MouseFocusReason);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelView::leaveEvent
 *****************************************************************************/

void PVParallelView::PVFullParallelView::leaveEvent(QEvent*)
{
	clearFocus();
}

/*****************************************************************************
 * PVParallelView::PVFullParallelView::fake_mouse_move
 *****************************************************************************/

void PVParallelView::PVFullParallelView::fake_mouse_move()
{
	QMouseEvent e((QEvent::MouseMove), mapFromGlobal(QCursor::pos()), QCursor::pos(), Qt::NoButton, Qt::NoButton,
	              Qt::NoModifier);
	QApplication::sendEvent(viewport(), &e);
}
