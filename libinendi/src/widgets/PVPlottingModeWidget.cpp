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

void PVWidgets::PVPlottingModeWidget::populate_from_type(QString const& type)
{
	LIB_CLASS(Inendi::PVPlottingFilter)
	::list_classes const& map_filters = LIB_CLASS(Inendi::PVPlottingFilter)::get().get_list();
	for (auto it = map_filters.begin(); it != map_filters.end(); it++) {
		Inendi::PVPlottingFilter::p_type filter = it->value();
		QString const& name = it->key();
		QString human_name = it->value()->get_human_name();
		_combo->addItem(human_name, name);
	}
}

void PVWidgets::PVPlottingModeWidget::populate_from_plotting(PVCol axis_id,
                                                             Inendi::PVPlotting& plotting)
{
	Inendi::PVPlottingProperties& props = plotting.get_properties_for_col(axis_id);
	QString type = plotting.get_plotted()
	                   ->get_parent<Inendi::PVSource>()
	                   .get_rushnraw()
	                   .collection()
	                   .formatter(axis_id)
	                   ->name();
	populate_from_type(type);
	set_mode(props.get_mode());
}
