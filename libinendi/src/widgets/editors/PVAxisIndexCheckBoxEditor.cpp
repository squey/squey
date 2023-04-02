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

#include <pvkernel/core/PVAxisIndexType.h>

#include <inendi/PVView.h>

#include <inendi/widgets/editors/PVAxisIndexCheckBoxEditor.h>

#include <QHBoxLayout>

/******************************************************************************
 *
 * PVCore::PVAxisIndexCheckBoxEditor::PVAxisIndexCheckBoxEditor
 *
 *****************************************************************************/
PVWidgets::PVAxisIndexCheckBoxEditor::PVAxisIndexCheckBoxEditor(Inendi::PVView const& view,
                                                                QWidget* parent)
    : QWidget(parent), _view(view)
{
	// _checked = true;	// Default is checked
	// _current_index = 0;

	auto layout = new QHBoxLayout;
	checkbox = new QCheckBox;
	checkbox->setCheckState(Qt::Checked);
	layout->addWidget(checkbox);
	// combobox = new QComboBox;
	// layout->addWidget(combobox);

	setLayout(layout);

	// connect(checkbox, SIGNAL(stateChanged(int)), this, SLOT(checkStateChanged_Slot(int)));

	setFocusPolicy(Qt::StrongFocus);
}

/******************************************************************************
 *
 * PVWidgets::PVAxisIndexCheckBoxEditor::~PVAxisIndexCheckBoxEditor
 *
 *****************************************************************************/
PVWidgets::PVAxisIndexCheckBoxEditor::~PVAxisIndexCheckBoxEditor() = default;

/******************************************************************************
 *
 * PVWidgets::PVAxisIndexCheckBoxEditor::set_axis_index
 *
 *****************************************************************************/
void PVWidgets::PVAxisIndexCheckBoxEditor::set_axis_index(
    PVCore::PVAxisIndexCheckBoxType /*axis_index*/)
{
	PVLOG_INFO("WE SET THE INDEX OF OUR CHECKBOX FROM THE EDITOR!\n");

	// combobox->clear();
	// combobox->addItems(_view.get_axes_names_list());
	// combobox->setCurrentIndex(axis_index.get_original_index());
}

PVCore::PVAxisIndexCheckBoxType PVWidgets::PVAxisIndexCheckBoxEditor::get_axis_index() const
{
	// 1 should be replace by the check to know if it is checked
	// return PVCore::PVAxisIndexCheckBoxType(currentIndex(), is_checked());
	PVLOG_INFO("WE GET THE INDEX OF CHECK BOX FROM THE EDITOR\n");

	return {(PVCol)combobox->currentIndex(), false};
}
