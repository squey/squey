/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVZoneIndexType.h>

#include <inendi/PVView.h>

#include <inendi/widgets/editors/PVZoneIndexEditor.h>

#include <QHBoxLayout>
#include <QLabel>

/******************************************************************************
 *
 * PVCore::PVZoneIndexEditor::PVZoneIndexEditor
 *
 *****************************************************************************/
PVWidgets::PVZoneIndexEditor::PVZoneIndexEditor(Inendi::PVView const& view, QWidget* parent)
    : QWidget(parent), _view(view)
{
	QHBoxLayout* hlayout = new QHBoxLayout();
	setLayout(hlayout);

	_first_cb = new QComboBox;
	_second_cb = new QComboBox;

	hlayout->addWidget(_first_cb);
	hlayout->addWidget(new QLabel("<->"));
	hlayout->addWidget(_second_cb);
}

/******************************************************************************
 *
 * PVWidgets::PVZoneIndexEditor::~PVZoneIndexEditor
 *
 *****************************************************************************/
PVWidgets::PVZoneIndexEditor::~PVZoneIndexEditor()
{
}

/******************************************************************************
 *
 * PVWidgets::PVZoneIndexEditor::set_zone_index
 *
 *****************************************************************************/
void PVWidgets::PVZoneIndexEditor::set_zone_index(PVCore::PVZoneIndexType zone_index)
{
	_first_cb->clear();
	_first_cb->addItems(_view.get_axes_names_list());
	_first_cb->setCurrentIndex(zone_index.get_zone_index_first());
	_second_cb->clear();
	_second_cb->addItems(_view.get_axes_names_list());
	_second_cb->setCurrentIndex(zone_index.get_zone_index_second());
}

PVCore::PVZoneIndexType PVWidgets::PVZoneIndexEditor::get_zone_index() const
{
	int index_first = _first_cb->currentIndex();
	int index_second = _second_cb->currentIndex();
	return PVCore::PVZoneIndexType(index_first, index_second);
}
