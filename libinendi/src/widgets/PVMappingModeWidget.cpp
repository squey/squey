/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVMapping.h>
#include <inendi/PVMapped.h>
#include <inendi/PVMappingFilter.h>
#include <inendi/PVView.h>

#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <inendi/widgets/PVArgumentListWidgetFactory.h>
#include <inendi/widgets/PVMappingModeWidget.h>

#include <QHBoxLayout>

PVWidgets::PVMappingModeWidget::PVMappingModeWidget(QWidget* parent) : QWidget(parent),
	_combo(new PVComboBox(this)),
	_props(nullptr)
  {
	QHBoxLayout* layout = new QHBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(10);
	layout->addWidget(_combo);
	setLayout(layout);

	setFocusPolicy(Qt::StrongFocus);
  }

PVWidgets::PVMappingModeWidget::PVMappingModeWidget(PVCol axis_id,
                                                    Inendi::PVMapping& mapping,
                                                    QWidget* parent)
    : PVMappingModeWidget(parent)
{
	populate_from_mapping(axis_id, mapping);
}

void PVWidgets::PVMappingModeWidget::populate_from_type(QString const& type)
{
	LIB_CLASS(Inendi::PVMappingFilter)
	::list_classes const& map_filters = LIB_CLASS(Inendi::PVMappingFilter)::get().get_list();
	LIB_CLASS(Inendi::PVMappingFilter)::list_classes::const_iterator it;
	for (it = map_filters.begin(); it != map_filters.end(); it++) {
		Inendi::PVMappingFilter::p_type filter = it->value();
		QString const& name = it->key();
		QString human_name = it->value()->get_human_name();
		QStringList params = name.split('_');
		if (params[0].compare(type) == 0) {
			_combo->addItem(human_name, params[1]);
		}
	}
	_cur_type = type;
}

void PVWidgets::PVMappingModeWidget::populate_from_mapping(PVCol axis_id,
                                                           Inendi::PVMapping& mapping)
{
	Inendi::PVMappingProperties& props = mapping.get_properties_for_col(axis_id);
	_props = &props;
	QString type = props.get_type();
	QString mode = props.get_mode();
	_filter_params[type][mode] = _props->get_args();
	populate_from_type(props.get_type());

	// FIXME : This is a hack as integer unsigned is a mapping while it should be a type.
	// This should be fix when mapping/plotting and type will be more standard.
	if (mode == "unsigned") {
		_combo->removeItem(_combo->findText("Signed decimal"));
	} else if (mode == "default" and type == "integer") {
		_combo->removeItem(_combo->findText("Unsigned decimal"));
	}

	set_mode(mode);
}

void PVWidgets::PVMappingModeWidget::set_filter_params_from_type_mode(QString const& type,
                                                                      QString const& mode)
{
	PVCore::PVArgumentList new_args;
	if (_filter_params.contains(type)) {
		new_args = _filter_params[type][mode];
	}

	if (new_args.size() == 0) {
		// Get default argument
		Inendi::PVMappingFilter::p_type lib_filter =
		    LIB_CLASS(Inendi::PVMappingFilter)::get().get_class_by_name(type + "_" + mode);
		new_args = lib_filter->get_default_args();
	}

	// Keep the argument that are the same from the previous args
	PVCore::PVArgumentList_set_common_args_from(new_args, _cur_filter_params);

	// And change them
	_cur_filter_params = new_args;
}

QSize PVWidgets::PVMappingModeWidget::sizeHint() const
{
	QLayout* l = layout();
	if (l) {
		return l->sizeHint();
	}
	return QSize();
}

void PVWidgets::PVMappingModeWidget::change_params()
{
	if (!_props) {
		return;
	}

	// Get argument from the properties and modify them
	PVCore::PVArgumentList args = _cur_filter_params;
	if (args.size() == 0) {
		return;
	}
	bool ret = PVWidgets::PVArgumentListWidget::modify_arguments_dlg(
	    PVWidgets::PVArgumentListWidgetFactory::create_mapping_plotting_widget_factory(), args,
	    this);
	if (!ret) {
		return;
	}

	_cur_filter_params = args;
	_filter_params[_cur_type][get_mode()] = args;
	_props->set_args(args);
}

bool PVWidgets::PVMappingModeWidget::set_mode(QString const& mode)
{
	return _combo->select_userdata(mode);
}
