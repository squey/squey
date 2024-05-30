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
#include <squey/PVScaled.h>
#include <squey/PVSource.h>

#include <squey/widgets/PVScalingModeWidget.h>

#include <QHBoxLayout>

PVWidgets::PVScalingModeWidget::PVScalingModeWidget(QWidget* parent)
    : QWidget(parent), _combo(new PVComboBox(this))
{
	auto* layout = new QHBoxLayout();
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(10);
	layout->addWidget(_combo);
	setLayout(layout);

	setFocusPolicy(Qt::StrongFocus);
}

PVWidgets::PVScalingModeWidget::PVScalingModeWidget(PVCol axis_id,
                                                      Squey::PVScaled& scaling,
                                                      QWidget* parent)
    : PVScalingModeWidget(parent)
{
	populate_from_scaling(axis_id, scaling);
}

QSize PVWidgets::PVScalingModeWidget::sizeHint() const
{
	QLayout* l = layout();
	if (l) {
		return l->sizeHint();
	}
	return {};
}

void PVWidgets::PVScalingModeWidget::populate_from_type(QString const& type, QString const& mapped)
{
	LIB_CLASS(Squey::PVScalingFilter)
	::list_classes const& map_filters = LIB_CLASS(Squey::PVScalingFilter)::get().get_list();
	for (const auto & map_filter : map_filters) {
		Squey::PVScalingFilter::p_type filter = map_filter.value();
		auto usable_type = filter->list_usable_type();
		if (usable_type.empty() or
		    std::find(usable_type.begin(), usable_type.end(),
		              std::make_pair(type.toStdString(), mapped.toStdString())) !=
		        usable_type.end()) {
			QString const& name = map_filter.key();
			QString human_name = filter->get_human_name();
			_combo->addItem(human_name, name);
		}
	}
}

void PVWidgets::PVScalingModeWidget::populate_from_scaling(PVCol axis_id,
                                                             Squey::PVScaled& scaling)
{
	Squey::PVScalingProperties& props = scaling.get_properties_for_col(axis_id);
	QString mapped =
	    QString::fromStdString(scaling.get_parent().get_properties_for_col(axis_id).get_mode());
	QString type =
	    scaling.get_parent<Squey::PVSource>().get_format().get_axes()[axis_id].get_type();
	populate_from_type(type, mapped);
	set_mode(QString::fromStdString(props.get_mode()));
}
