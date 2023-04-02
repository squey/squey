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

#include <inendi/PVMapped.h>
#include <inendi/PVSource.h>
#include <inendi/PVMappingFilter.h>

#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <inendi/widgets/PVArgumentListWidgetFactory.h>
#include <inendi/widgets/PVMappingModeWidget.h>

#include <QHBoxLayout>

PVWidgets::PVMappingModeWidget::PVMappingModeWidget(QWidget* parent)
    : QWidget(parent), _combo(new PVComboBox(this))
{
	QHBoxLayout* layout = new QHBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(10);
	layout->addWidget(_combo);
	setLayout(layout);

	setFocusPolicy(Qt::StrongFocus);
}

PVWidgets::PVMappingModeWidget::PVMappingModeWidget(PVCol axis_id,
                                                    Inendi::PVMapped& mapping,
                                                    QWidget* parent)
    : PVMappingModeWidget(parent)
{
	populate_from_mapping(axis_id, mapping);
}

void PVWidgets::PVMappingModeWidget::populate_from_type(QString const& type)
{
	LIB_CLASS(Inendi::PVMappingFilter)
	::list_classes const& map_filters = LIB_CLASS(Inendi::PVMappingFilter)::get().get_list();
	for (const auto & map_filter : map_filters) {
		Inendi::PVMappingFilter::p_type filter = map_filter.value();
		auto available_type = filter->list_usable_type();
		if (available_type.find(type.toStdString()) != available_type.end()) {
			_combo->addItem(filter->get_human_name(), map_filter.key());
		}
	}
}

void PVWidgets::PVMappingModeWidget::populate_from_mapping(PVCol axis_id, Inendi::PVMapped& mapping)
{
	Inendi::PVMappingProperties& props = mapping.get_properties_for_col(axis_id);
	QString type = mapping.get_parent().get_format().get_axes()[axis_id].get_type();
	QString mode = QString::fromStdString(props.get_mode());
	populate_from_type(type);

	set_mode(mode);
}

QSize PVWidgets::PVMappingModeWidget::sizeHint() const
{
	QLayout* l = layout();
	if (l) {
		return l->sizeHint();
	}
	return QSize();
}

bool PVWidgets::PVMappingModeWidget::set_mode(QString const& mode)
{
	return _combo->select_userdata(mode);
}
