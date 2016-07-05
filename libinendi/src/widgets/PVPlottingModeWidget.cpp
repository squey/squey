/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVPlotting.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVSource.h>

#include <inendi/widgets/PVPlottingModeWidget.h>

#include <QHBoxLayout>

PVWidgets::PVPlottingModeWidget::PVPlottingModeWidget(QWidget* parent)
    : QWidget(parent), _combo(new PVComboBox(this))
{
	QHBoxLayout* layout = new QHBoxLayout();
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(10);
	layout->addWidget(_combo);
	setLayout(layout);

	setFocusPolicy(Qt::StrongFocus);
}

PVWidgets::PVPlottingModeWidget::PVPlottingModeWidget(PVCol axis_id,
                                                      Inendi::PVPlotting& plotting,
                                                      QWidget* parent)
    : PVPlottingModeWidget(parent)
{
	populate_from_plotting(axis_id, plotting);
}

QSize PVWidgets::PVPlottingModeWidget::sizeHint() const
{
	QLayout* l = layout();
	if (l) {
		return l->sizeHint();
	}
	return QSize();
}

void PVWidgets::PVPlottingModeWidget::populate_from_type(QString const& type, QString const& mapped)
{
	LIB_CLASS(Inendi::PVPlottingFilter)
	::list_classes const& map_filters = LIB_CLASS(Inendi::PVPlottingFilter)::get().get_list();
	for (auto it = map_filters.begin(); it != map_filters.end(); it++) {
		Inendi::PVPlottingFilter::p_type filter = it->value();
		auto usable_type = filter->list_usable_type();
		if (usable_type.empty() or
		    std::find(usable_type.begin(), usable_type.end(),
		              std::make_pair(type.toStdString(), mapped.toStdString())) !=
		        usable_type.end()) {
			QString const& name = it->key();
			QString human_name = filter->get_human_name();
			_combo->addItem(human_name, name);
		}
	}
}

void PVWidgets::PVPlottingModeWidget::populate_from_plotting(PVCol axis_id,
                                                             Inendi::PVPlotting& plotting)
{
	Inendi::PVPlottingProperties& props = plotting.get_properties_for_col(axis_id);
	QString mapped = plotting.get_plotted()
	                     ->get_parent()
	                     .get_mapping()
	                     .get_properties_for_col(axis_id)
	                     .get_mode();
	QString type = plotting.get_plotted()
	                   ->get_parent<Inendi::PVSource>()
	                   .get_format()
	                   .get_axes()[axis_id]
	                   .get_type();
	populate_from_type(type, mapped);
	set_mode(props.get_mode());
}
