/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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
	for (auto it = map_filters.begin(); it != map_filters.end(); it++) {
		Inendi::PVMappingFilter::p_type filter = it->value();
		auto available_type = filter->list_usable_type();
		if (available_type.find(type.toStdString()) != available_type.end()) {
			_combo->addItem(filter->get_human_name(), it->key());
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
