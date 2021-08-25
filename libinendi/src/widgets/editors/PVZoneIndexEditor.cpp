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
