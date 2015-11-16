/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVSpinBoxType.h>

#include <inendi/PVView.h>
#include <inendi/widgets/editors/PVViewRowsSpinBoxEditor.h>

/******************************************************************************
 *
 * PVCore::PVViewRowsSpinBoxEditor::PVViewRowsSpinBoxEditor
 *
 *****************************************************************************/
PVWidgets::PVViewRowsSpinBoxEditor::PVViewRowsSpinBoxEditor(Inendi::PVView const& view, QWidget *parent):
	QSpinBox(parent),
	_view(view)
{
}

/******************************************************************************
 *
 * PVWidgets::PVViewRowsSpinBoxEditor::~PVViewRowsSpinBoxEditor
 *
 *****************************************************************************/
PVWidgets::PVViewRowsSpinBoxEditor::~PVViewRowsSpinBoxEditor()
{
}

/******************************************************************************
 *
 * PVWidgets::PVViewRowsSpinBoxEditor::set_spin
 *
 *****************************************************************************/
void PVWidgets::PVViewRowsSpinBoxEditor::set_spin(PVCore::PVSpinBoxType s)
{
	setRange(1, _view.get_row_count());
	setValue(s.get_value());
}

/******************************************************************************
 *
 * PVWidgets::PVViewRowsSpinBoxEditor::get_spin
 *
 *****************************************************************************/
PVCore::PVSpinBoxType PVWidgets::PVViewRowsSpinBoxEditor::get_spin() const
{
	PVCore::PVSpinBoxType ret(value());
	return ret;
}
