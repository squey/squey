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

#include <pvparallelview/PVFullParallelViewParamsWidget.h>
#include <pvparallelview/PVFullParallelViewSelectionRectangle.h>
#include <pvparallelview/PVFullParallelView.h>
#include <pvparallelview/PVFullParallelScene.h>

#include <pvkernel/widgets/PVModdedIcon.h>
#include <pvparallelview/PVAntialiasingAction.h>

#include <squey/PVScaled.h>

#include <QVBoxLayout>
#include <QToolBar>
#include <QCheckBox>
#include <QSignalBlocker>
#include <QSignalMapper>
#include <QMenu>
#include <QLineEdit>
#include <QLabel>
#include <QLocale>
#include <QSlider>
#include <QToolButton>
#include <QDebug>

#include <algorithm>
#include <cmath>

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::PVFullParallelViewParamsWidget
 *****************************************************************************/

PVParallelView::PVFullParallelViewParamsWidget::PVFullParallelViewParamsWidget(
    PVFullParallelView* parent)
    : PVFloatingToolBar(parent)
{
	auto density_action = addAction(QIcon(":/density-axis"), "Density on axes");
	density_action->setCheckable(true);
	density_action->setChecked(false);
	density_action->setShortcut(Qt::Key_D);
	QImage density_legend(60, 1, QImage::Format_ARGB32);
	for (int i = 0; i < density_legend.width(); ++i) {
		density_legend.setPixelColor(
		    i, 0, QColor::fromHsvF((1. - double(i) / density_legend.width()) * 2 / 3., 1., 1.));
	}
	auto density_legend_label = new QLabel();
	density_legend_label->setPixmap(
	    QPixmap::fromImage(density_legend).scaled(density_legend.width(), 16));
	auto dll_action = addWidget(density_legend_label);
	dll_action->setVisible(false);
	connect(density_action, &QAction::toggled, [this, dll_action](bool pushed) {
		auto scene = static_cast<PVParallelView::PVFullParallelScene*>(parent_fpv()->scene());
		scene->get_lines_view().set_axis_width(pushed ? 21 : 3);
		scene->enable_density_on_axes(pushed);
		scene->update_number_of_zones_async();
		dll_action->setVisible(pushed);
		adjustSize();
	});

	_lines_by_density = addAction(PVModdedIcon("density-lines"), tr("Lines by density"));
	_lines_by_density->setCheckable(true);
	_lines_by_density->setToolTip(
	    tr("Lines by density: the more rows a line stands for, the more opaque it is"));
	// Tenths of a decade of the opacity of the line of a single row.
	_line_opacity_slider = new QSlider(Qt::Horizontal);
	_line_opacity_slider->setRange(-70, -1);
	// The most opaque to start with.
	_line_opacity_slider->setValue(_line_opacity_slider->maximum());
	_line_opacity_slider->setFixedWidth(120);
	auto line_opacity_action = addWidget(_line_opacity_slider);
	line_opacity_action->setVisible(false);
	connect(_lines_by_density, &QAction::toggled, [this, line_opacity_action](bool pushed) {
		auto* s = scene();
		if (s == nullptr) {
			return;
		}
		update_line_opacity_tooltip();
		line_opacity_action->setVisible(pushed);
		s->set_line_opacity(pushed ? line_opacity() : 1.f);
		adjustSize();
	});
	connect(_line_opacity_slider, &QSlider::valueChanged, [this]() {
		update_line_opacity_tooltip();
		if (auto* s = scene(); s != nullptr and _lines_by_density->isChecked()) {
			s->set_line_opacity(line_opacity());
		}
	});

	auto* antialiased = new PVAntialiasingAction(this);
	addAction(antialiased);
	connect(antialiased, &QAction::toggled, [this](bool pushed) {
		if (auto* s = scene()) {
			s->set_antialiased(pushed);
		}
	});

	// Selection scaling: the button turns it on for every axis, its menu rescales
	// once on demand and follows the selection on its own. Both are held on the
	// model, so an axis switched on its own through its header menu keeps its say.
	_scale_on_selection_button = new QToolButton(this);
	// A themed icon, so that it is legible under both colour schemes: a plain
	// QIcon on a resource file carries one rendering and stays dark on dark.
	_scale_on_selection_button->setIcon(PVModdedIcon("scaling"));
	// The size the toolbar gives its actions: a widget added to a toolbar does not
	// inherit it, and the button would otherwise carry an icon smaller than the
	// one beside it.
	_scale_on_selection_button->setIconSize(iconSize());
	_scale_on_selection_button->setCheckable(true);
	_scale_on_selection_button->setPopupMode(QToolButton::MenuButtonPopup);
	_scale_on_selection_button->setToolTip(tr("Selection stretch"));

	auto* scale_menu = new QMenu(_scale_on_selection_button);
	scale_menu->setAttribute(Qt::WA_TranslucentBackground);
	_rescale_now = scale_menu->addAction(tr("Stretch on current selection"));
	_auto_rescale = scale_menu->addAction(tr("Stretch on each selection"));
	_auto_rescale->setCheckable(true);
	_scale_on_selection_button->setMenu(scale_menu);
	addWidget(_scale_on_selection_button);

	connect(_scale_on_selection_button, &QToolButton::toggled, [this](bool pushed) {
		if (auto* s = scene()) {
			s->set_scale_on_selection(pushed);
		}
	});
	connect(_rescale_now, &QAction::triggered, [this]() {
		if (auto* s = scene()) {
			s->rescale_on_selection();
		}
	});
	connect(_auto_rescale, &QAction::toggled, [this](bool pushed) {
		if (auto* s = scene()) {
			s->set_auto_scale_on_selection(pushed);
		}
	});

	setVisible(true);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::update_widgets
 *****************************************************************************/

void PVParallelView::PVFullParallelViewParamsWidget::update_widgets()
{
	auto* s = scene();
	if (s == nullptr) {
		return;
	}

	const Squey::PVScaled& scaled = s->lib_view().get_parent<Squey::PVScaled>();

	// Set without going back through the handlers, which would ask the model to
	// do again what it is already doing.
	QSignalBlocker block_button(_scale_on_selection_button);
	QSignalBlocker block_auto(_auto_rescale);
	_scale_on_selection_button->setChecked(scaled.scale_on_selection());
	_auto_rescale->setChecked(scaled.auto_scale_on_selection());
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::scene
 *****************************************************************************/

PVParallelView::PVFullParallelScene* PVParallelView::PVFullParallelViewParamsWidget::scene() const
{
	return static_cast<PVParallelView::PVFullParallelScene*>(parent_fpv()->scene());
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::line_opacity
 *****************************************************************************/

float PVParallelView::PVFullParallelViewParamsWidget::line_opacity() const
{
	return std::pow(10.f, _line_opacity_slider->value() / 10.f);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::update_line_opacity_tooltip
 *****************************************************************************/

void PVParallelView::PVFullParallelViewParamsWidget::update_line_opacity_tooltip()
{
	// 1 - (1 - opacity)^rows = 1/2
	const double rows = std::log(.5) / std::log1p(-double(line_opacity()));
	_line_opacity_slider->setToolTip(
	    tr("A line of %1 row(s) is half opaque")
	        .arg(QLocale(QLocale::English).toString(qlonglong(std::max(1., std::round(rows))))));
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::set_selection_mode
 *****************************************************************************/

void PVParallelView::PVFullParallelViewParamsWidget::set_selection_mode(int mode)
{
	PVSelectionRectangle::update_selection_mode_selector(_sel_mode_button, mode);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::parent_hcv
 *****************************************************************************/

PVParallelView::PVFullParallelView*
PVParallelView::PVFullParallelViewParamsWidget::parent_fpv() const
{
	assert(qobject_cast<PVFullParallelView*>(parentWidget()));
	return static_cast<PVFullParallelView*>(parentWidget());
}
