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

#include <pvparallelview/PVScatterViewParamsWidget.h>
#include <pvparallelview/PVScatterViewSelectionRectangle.h>
#include <pvparallelview/PVScatterView.h>

#include <QSignalMapper>
#include <QToolBar>
#include <QToolButton>
#include <QAction>

/*****************************************************************************
 * PVParallelView::PVScatterViewParamsWidget::PVScatterViewParamsWidget
 *****************************************************************************/

PVParallelView::PVScatterViewParamsWidget::PVScatterViewParamsWidget(PVScatterView* parent)
    : QToolBar(parent)
{
	_sel_mode_signal_mapper = new QSignalMapper(this);
	QObject::connect(_sel_mode_signal_mapper,
	                 &QSignalMapper::mappedInt, this,
	                 &PVScatterViewParamsWidget::set_selection_mode);

	_sel_mode_button =
	    PVSelectionRectangle::add_selection_mode_selector(parent, this, _sel_mode_signal_mapper);

	_show_labels = new QAction(this);
	_show_labels->setIcon(QIcon(":/labeled-axis"));
	_show_labels->setCheckable(true);
	_show_labels->setChecked(false);
	_show_labels->setShortcut(Qt::Key_T);
	_show_labels->setText("Toggle labels visibility");
	_show_labels->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	_show_labels->setToolTip("Activate/deactivate labels display on axes (" +
	                         _show_labels->shortcut().toString() + ")");
	addAction(_show_labels);
	parent->addAction(_show_labels);
	connect(_show_labels, &QAction::toggled, parent_sv(), &PVScatterView::toggle_show_labels);
}

/*****************************************************************************
 * PVParallelView::PVScatterViewParamsWidget::update_widgets
 *****************************************************************************/

void PVParallelView::PVScatterViewParamsWidget::update_widgets()
{
	_show_labels->blockSignals(true);

	_show_labels->setChecked(parent_sv()->show_labels());

	_show_labels->blockSignals(false);
}

/*****************************************************************************
 * PVParallelView::PVScatterViewParamsWidget::set_selection_mode
 *****************************************************************************/

void PVParallelView::PVScatterViewParamsWidget::set_selection_mode(int mode)
{
	PVSelectionRectangle::update_selection_mode_selector(_sel_mode_button, mode);

	PVScatterView* sv = parent_sv();
	sv->get_selection_rect()->set_selection_mode(mode);
	sv->fake_mouse_move();
	sv->get_viewport()->update();
}

/*****************************************************************************
 * PVParallelView::PVScatterViewParamsWidget::parent_sv
 *****************************************************************************/

PVParallelView::PVScatterView* PVParallelView::PVScatterViewParamsWidget::parent_sv()
{
	assert(qobject_cast<PVScatterView*>(parent()));
	return static_cast<PVScatterView*>(parent());
}
