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

#include <pvparallelview/PVHitCountViewParamsWidget.h>
#include <pvparallelview/PVHitCountViewSelectionRectangle.h>
#include <pvparallelview/PVHitCountView.h>

#include <QVBoxLayout>
#include <QToolBar>
#include <QCheckBox>
#include <QSignalMapper>
#include <QMenu>

/*****************************************************************************
 * PVParallelView::PVHitCountViewParamsWidget::PVHitCountViewParamsWidget
 *****************************************************************************/

PVParallelView::PVHitCountViewParamsWidget::PVHitCountViewParamsWidget(PVHitCountView* parent)
    : QToolBar(parent)
{
	_signal_mapper = new QSignalMapper(this);
	QObject::connect(_signal_mapper,
	                 &QSignalMapper::mappedInt, this,
	                 &PVHitCountViewParamsWidget::set_selection_mode);

	_sel_mode_button =
	    PVSelectionRectangle::add_selection_mode_selector(parent, this, _signal_mapper);

	addSeparator();

	_autofit = new QAction(this);
	_autofit->setIcon(QIcon(":/zoom-autofit-horizontal"));
	_autofit->setCheckable(true);
	_autofit->setChecked(false);
	_autofit->setShortcut(Qt::Key_F);
	_autofit->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	_autofit->setText("View auto-fit on selected events");
	_autofit->setToolTip("Activate/deactivate horizontal auto-fit on selected events (" +
	                     _autofit->shortcut().toString() + ")");
	addAction(_autofit);
	parent->addAction(_autofit);
	connect(_autofit, &QAction::toggled, parent_hcv(), &PVHitCountView::toggle_auto_x_zoom_sel);

	_use_log_color = new QAction("Logarithmic colormap", this);
	_use_log_color->setIcon(QIcon(":/colormap-log"));
	_use_log_color->setCheckable(true);
	_use_log_color->setChecked(false);
	_use_log_color->setShortcut(Qt::Key_L);
	_use_log_color->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	_use_log_color->setText("Logarithmic colormap");
	_use_log_color->setToolTip(
	    "Activate/deactivate use of a logarithmic colormap for visible events (" +
	    _use_log_color->shortcut().toString() + ")");
	addAction(_use_log_color);
	parent->addAction(_use_log_color);
	connect(_use_log_color, &QAction::toggled, parent_hcv(), &PVHitCountView::toggle_log_color);

	_show_labels = new QAction(this);
	_show_labels->setIcon(QIcon(":/labeled-axis"));
	_show_labels->setCheckable(true);
	_show_labels->setChecked(false);
	_show_labels->setShortcut(Qt::Key_T);
	_show_labels->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	_show_labels->setText("Toggle labels visibility");
	_show_labels->setToolTip("Activate/deactivate labels display on axes (" +
	                         _show_labels->shortcut().toString() + ")");
	addAction(_show_labels);
	parent->addAction(_show_labels);
	connect(_show_labels, &QAction::toggled, parent_hcv(), &PVHitCountView::toggle_show_labels);
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewParamsWidget::update_widgets
 *****************************************************************************/

void PVParallelView::PVHitCountViewParamsWidget::update_widgets()
{
	_autofit->blockSignals(true);
	_use_log_color->blockSignals(true);
	_show_labels->blockSignals(true);

	_autofit->setChecked(parent_hcv()->auto_x_zoom_sel());
	_use_log_color->setChecked(parent_hcv()->use_log_color());
	_show_labels->setChecked(parent_hcv()->show_labels());

	_autofit->blockSignals(false);
	_use_log_color->blockSignals(false);
	_show_labels->blockSignals(false);
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewParamsWidget::set_selection_mode
 *****************************************************************************/

void PVParallelView::PVHitCountViewParamsWidget::set_selection_mode(int mode)
{
	PVHitCountView* hcv = parent_hcv();
	hcv->get_selection_rect()->set_selection_mode(mode);
	hcv->fake_mouse_move();
	hcv->get_viewport()->update();

	PVSelectionRectangle::update_selection_mode_selector(_sel_mode_button, mode);
}

/*****************************************************************************
 * PVParallelView::PVHitCountViewParamsWidget::parent_hcv
 *****************************************************************************/

PVParallelView::PVHitCountView* PVParallelView::PVHitCountViewParamsWidget::parent_hcv()
{
	assert(qobject_cast<PVHitCountView*>(parentWidget()));
	return static_cast<PVHitCountView*>(parentWidget());
}
