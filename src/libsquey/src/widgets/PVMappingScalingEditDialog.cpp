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

#include <squey/PVScaled.h>
#include <squey/PVSource.h>

#include <squey/widgets/PVMappingModeWidget.h>
#include <squey/widgets/PVScalingModeWidget.h>
#include <squey/widgets/PVMappingScalingEditDialog.h>

#include <QDialogButtonBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QScrollArea>

static QList<PVRush::PVAxisFormat> const& init_axes(Squey::PVMapped* mapping,
                                                    Squey::PVScaled* scaling)
{
	if (mapping) {
		return mapping->get_parent<Squey::PVSource>().get_format().get_axes();
	}

	assert(scaling);
	return scaling->get_parent<Squey::PVSource>().get_format().get_axes();
}

/******************************************************************************
 *
 * Squey::PVMappingScalingEditDialog::PVMappingScalingEditDialog
 *
 *****************************************************************************/
PVWidgets::PVMappingScalingEditDialog::PVMappingScalingEditDialog(Squey::PVMapped* mapping,
                                                                    Squey::PVScaled* scaling,
                                                                    QWidget* parent)
    : QDialog(parent), _mapping(mapping), _scaling(scaling), _axes(init_axes(_mapping, _scaling))
{
	PVLOG_DEBUG("PVWidgets::PVMappingScalingEditDialog::%s\n", __FUNCTION__);

#ifndef NDEBUG
	if (has_mapping() && has_scaling()) {
		assert(&_mapping->get_parent<Squey::PVSource>() ==
		       &_scaling->get_parent<Squey::PVSource>());
	} else {
		assert(has_mapping() || has_scaling());
	}
#endif

	setWindowTitle(tr("Edit properties..."));

	// We generate and populate the Layout
	init_layout();
	// We load the current settings
	load_settings();
	// We can set the layout in the Widget
	finish_layout();

	// Adapt scroll area width so that everything fits in (closes #0x100)
	setMinimumWidth(_main_scroll_area->viewport()->width());
}

/******************************************************************************
 *
 * PVWidgets::PVMappingScalingEditDialog::create_label
 *
 *****************************************************************************/
QLabel* PVWidgets::PVMappingScalingEditDialog::create_label(QString const& text,
                                                             Qt::Alignment align)
{
	PVLOG_DEBUG("PVWidgets::PVMappingScalingEditDialog::%s\n", __FUNCTION__);

	auto* ret = new QLabel(text, nullptr);
	ret->setAlignment(align);
	QFont font(ret->font());
	font.setBold(true);
	ret->setFont(font);
	return ret;
}

/******************************************************************************
 *
 * PVWidgets::PVMappingScalingEditDialog::finish_layout
 *
 *****************************************************************************/
void PVWidgets::PVMappingScalingEditDialog::finish_layout()
{
	PVLOG_DEBUG("PVWidgets::PVMappingScalingEditDialog::%s\n", __FUNCTION__);

	auto* btns = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel | QDialogButtonBox::Apply);
	connect(btns, &QDialogButtonBox::clicked, [this, btns](auto* button){
		switch (btns->buttonRole(button))
		{
		case QDialogButtonBox::AcceptRole:
			save_settings();
			accept();
			break;
		case QDialogButtonBox::ApplyRole:
			save_settings();
			break;
		case QDialogButtonBox::RejectRole:
			reject();
			break;
		
		default:
			break;
		}
	});
	_main_layout->addWidget(btns);
}

/******************************************************************************
 *
 * PVWidgets::PVMappingScalingEditDialog::init_layout
 *
 *****************************************************************************/
void PVWidgets::PVMappingScalingEditDialog::init_layout()
{
	PVLOG_DEBUG("PVWidgets::PVMappingScalingEditDialog::%s\n", __FUNCTION__);

	_main_layout = new QVBoxLayout();
	_main_layout->setSpacing(0);
	_main_layout->setContentsMargins(0, 0, 0, 0);

	auto* grid_widget = new QWidget();
	_main_grid = new QGridLayout(grid_widget);
	_main_grid->setHorizontalSpacing(10);
	_main_grid->setVerticalSpacing(10);
	int row = 0;
	int col = 0;

	// Init titles
	_main_grid->addWidget(create_label(tr("Axis"), Qt::AlignLeft), row, col);
	_main_grid->setColumnStretch(col, 1);
	col++;
	if (has_mapping()) {
		_main_grid->addWidget(create_label(tr("Type")), row, col);
		_main_grid->setColumnStretch(col, 1);
		col++;
		_main_grid->addWidget(create_label(tr("Mapping")), row, col);
		_main_grid->setColumnStretch(col, 2);
		col++;
	}
	if (has_scaling()) {
		_main_grid->addWidget(create_label(tr("Scaling")), row, col);
		_main_grid->setColumnStretch(col, 2);
		col++;
	}
	_main_scroll_area = new QScrollArea();
	_main_scroll_area->setWidget(grid_widget);
	_main_scroll_area->setWidgetResizable(true);

	_main_layout->addWidget(_main_scroll_area);

	setLayout(_main_layout);
}

/******************************************************************************
 *
 * PVWidgets::PVMappingScalingEditDialog::load_settings
 *
 *****************************************************************************/
void PVWidgets::PVMappingScalingEditDialog::load_settings()
{
	PVLOG_DEBUG("PVWidgets::PVMappingScalingEditDialog::%s\n", __FUNCTION__);

	// Add widgets

	int row = 1;
	for (PVRush::PVAxisFormat const& axe : _axes) {
		int col = 0;
		_main_grid->addWidget(new QLabel(axe.get_name(), this), row, col++);
		if (has_mapping()) {
			_main_grid->addWidget(new QLabel(axe.get_type()), row, col++);
			_main_grid->addWidget(new PVWidgets::PVMappingModeWidget(axe.index, *_mapping, this),
			                      row, col++);
		}
		if (has_scaling()) {
			_main_grid->addWidget(new PVWidgets::PVScalingModeWidget(axe.index, *_scaling, this),
			                      row, col++);
		}
		row++;
	}
}

/******************************************************************************
 *
 * PVWidgets::PVMappingScalingEditDialog::save_settings
 *
 *****************************************************************************/
void PVWidgets::PVMappingScalingEditDialog::save_settings()
{
	PVLOG_DEBUG("PVWidgets::PVMappingScalingEditDialog::%s\n", __FUNCTION__);

	int row = 1;
	for (PVRush::PVAxisFormat const& axis : _axes) {
		int col = 0;
		if (has_mapping()) {
			QString type = axis.get_type();
			Squey::PVMappingProperties& prop = _mapping->get_properties_for_col(axis.index);

			// Mapping mode
			auto* map_combo =
			    dynamic_cast<PVWidgets::PVMappingModeWidget*>(
			        _main_grid->itemAtPosition(row, col += 2)->widget());
			assert(map_combo);
			QString mode = map_combo->get_mode();

			prop.set_mode(mode.toStdString());
		}
		if (has_scaling()) {
			auto* combo = dynamic_cast<PVWidgets::PVScalingModeWidget*>(
			    _main_grid->itemAtPosition(row, col += 1)->widget());
			assert(combo);
			QString mode = combo->get_mode();
			_scaling->get_properties_for_col(axis.index).set_mode(mode.toStdString());
		}
		row++;
	}

	if (_mapping && !_mapping->is_uptodate()) {
		_mapping->update_mapping();
	}
	if (_scaling && !_scaling->is_uptodate()) {
		_scaling->update_scaling();
	}
}
