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

#include <squey/PVView.h>

#include <squey/widgets/editors/PVAxisIndexEditor.h>

/******************************************************************************
 *
 * PVCore::PVAxisIndexEditor::PVAxisIndexEditor
 *
 *****************************************************************************/
PVWidgets::PVAxisIndexEditor::PVAxisIndexEditor(Squey::PVView const& view, QWidget* parent)
    : QComboBox(parent), _view(view)
{
}

/******************************************************************************
 *
 * PVWidgets::PVAxisIndexEditor::set_axis_index
 *
 *****************************************************************************/
void PVWidgets::PVAxisIndexEditor::set_axis_index(PVCore::PVAxisIndexType axis_index)
{
	clear();
	addItems(_view.get_axes_names_list());
	setCurrentIndex(axis_index.get_axis_index());
}

PVCore::PVAxisIndexType PVWidgets::PVAxisIndexEditor::get_axis_index() const
{
	PVCombCol comb_col(currentIndex());
	PVCol index = _view.get_axes_combination().get_nraw_axis(comb_col);
	return PVCore::PVAxisIndexType(index, false, comb_col);
}
