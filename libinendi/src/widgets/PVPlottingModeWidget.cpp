/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVPlotting.h>
#include <inendi/PVView.h>

#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <inendi/widgets/PVArgumentListWidgetFactory.h>
#include <inendi/widgets/PVPlottingModeWidget.h>

#include <QHBoxLayout>

PVWidgets::PVPlottingModeWidget::PVPlottingModeWidget(QString const& type, QWidget* parent)
    : QWidget(parent)
{
	init(false);
	populate_from_type(type);
}

PVWidgets::PVPlottingModeWidget::PVPlottingModeWidget(PVCol axis_id,
                                                      Inendi::PVPlotting& plotting,
                                                      bool params_btn,
                                                      QWidget* parent)
    : QWidget(parent)
{
	init(params_btn);
	populate_from_plotting(axis_id, plotting);
}

PVWidgets::PVPlottingModeWidget::PVPlottingModeWidget(PVCol axis_id,
                                                      Inendi::PVView& view,
                                                      bool params_btn,
                                                      QWidget* parent)
    : QWidget(parent)
{
	init(params_btn);
	populate_from_plotting(axis_id, view.get_parent<Inendi::PVPlotted>()->get_plotting());
}

QSize PVWidgets::PVPlottingModeWidget::sizeHint() const
{
	QLayout* l = layout();
	if (l) {
		return l->sizeHint();
	}
	return QSize();
}

void PVWidgets::PVPlottingModeWidget::init(bool params_btn)
{
	_combo = new PVComboBox(this);
	_props = NULL;

	QHBoxLayout* layout = new QHBoxLayout();
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(10);
	layout->addWidget(_combo);
	if (params_btn) {
		_params_btn = new QPushButton(tr("Parameters..."));
		QSizePolicy sp(QSizePolicy::Maximum, QSizePolicy::Fixed);
		_params_btn->setSizePolicy(sp);
		layout->addWidget(_params_btn);

		connect(_params_btn, SIGNAL(clicked()), this, SLOT(change_params()));
	} else {
		_params_btn = NULL;
	}
	setLayout(layout);

	setFocusPolicy(Qt::StrongFocus);
}

void PVWidgets::PVPlottingModeWidget::populate_from_type(QString const& type)
{
	LIB_CLASS(Inendi::PVPlottingFilter)
	::list_classes const& map_filters = LIB_CLASS(Inendi::PVPlottingFilter)::get().get_list();
	LIB_CLASS(Inendi::PVPlottingFilter)::list_classes::const_iterator it;
	for (it = map_filters.begin(); it != map_filters.end(); it++) {
		Inendi::PVPlottingFilter::p_type filter = it->value();
		QString const& name = it->key();
		QString human_name = it->value()->get_human_name();
		QStringList params = name.split('_');
		if (params[0].compare(type) == 0) {
			_combo->addItem(human_name, params[1]);
		}
	}
}

void PVWidgets::PVPlottingModeWidget::populate_from_plotting(PVCol axis_id,
                                                             Inendi::PVPlotting& plotting)
{
	Inendi::PVPlottingProperties& props = plotting.get_properties_for_col(axis_id);
	_props = &props;
	populate_from_type(props.get_type());
	set_mode(props.get_mode());

	if (_props->get_args().size() == 0 && _params_btn) {
		_params_btn->setEnabled(false);
	}
}

void PVWidgets::PVPlottingModeWidget::change_params()
{
	if (!_props) {
		return;
	}

	// Get argument from the properties and modify them
	PVCore::PVArgumentList args = _props->get_args();
	if (args.size() == 0) {
		return;
	}
	bool ret = PVWidgets::PVArgumentListWidget::modify_arguments_dlg(
	    PVWidgets::PVArgumentListWidgetFactory::create_mapping_plotting_widget_factory(), args,
	    this);
	if (!ret) {
		return;
	}

	_props->set_args(args);
}
