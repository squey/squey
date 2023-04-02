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

#include <pvkernel/core/PVSpinBoxType.h>

#include <inendi/PVView.h>
#include <inendi/widgets/editors/PVViewRowsSpinBoxEditor.h>

/******************************************************************************
 *
 * PVCore::PVViewRowsSpinBoxEditor::PVViewRowsSpinBoxEditor
 *
 *****************************************************************************/
PVWidgets::PVViewRowsSpinBoxEditor::PVViewRowsSpinBoxEditor(Inendi::PVView const& view,
                                                            QWidget* parent)
    : QSpinBox(parent), _view(view)
{
}

/******************************************************************************
 *
 * PVWidgets::PVViewRowsSpinBoxEditor::~PVViewRowsSpinBoxEditor
 *
 *****************************************************************************/
PVWidgets::PVViewRowsSpinBoxEditor::~PVViewRowsSpinBoxEditor()
= default;

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
	PVCore::PVSpinBoxType ret(PVCol((PVCol::value_type)value()));
	return ret;
}
