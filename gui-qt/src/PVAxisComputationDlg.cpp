#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <picviz/PVAxisComputation.h>
#include <picviz/widgets/PVArgumentListWidgetFactory.h>

#include <PVAxisComputationDlg.h>

PVInspector::PVAxisComputationDlg::PVAxisComputationDlg(Picviz::PVView& view, QWidget* parent):
	QDialog(parent)
{
	setupUi(this);
	_args_plugin_widget->set_widget_factory(PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(view));

	init_plugins(_combo_plugins);
	update_plugin_args();

	connect(_combo_plugins, SIGNAL(currentIndexChanged(int)), this, SLOT(update_plugin_args()));
}

void PVInspector::PVAxisComputationDlg::init_plugins(QComboBox* cb)
{
	// List all available plugins and add them to the combo box
	LIB_CLASS(Picviz::PVAxisComputation)::list_classes const& axis_plugins = LIB_CLASS(Picviz::PVAxisComputation)::get().get_list();
	LIB_CLASS(Picviz::PVAxisComputation)::list_classes::const_iterator it;
	for (it = axis_plugins.begin(); it != axis_plugins.end(); it++) {
		QString name = it.value()->get_human_name();
		QString key = it.key();
		cb->addItem(name, key);
	}
	cb->setCurrentIndex(0);
}

void PVInspector::PVAxisComputationDlg::update_plugin_args()
{
	int cur_idx = _combo_plugins->currentIndex();
	if (cur_idx < 0) {
		return;
	}

	QString cur_plugin = _combo_plugins->itemData(cur_idx).toString();
	Picviz::PVAxisComputation_p plugin_lib = LIB_CLASS(Picviz::PVAxisComputation)::get().get_class_by_name(cur_plugin);
	if (!plugin_lib) {
		PVLOG_ERROR("(PVInspector::PVAxisComputationDlg) unable to find plugin %s\n", qPrintable(cur_plugin));
		return;
	}
	_plugin_args = plugin_lib->get_default_args();

	_cur_plugin = plugin_lib->clone<Picviz::PVAxisComputation>();
	_args_plugin_widget->set_args(_plugin_args);
}

Picviz::PVAxisComputation_p PVInspector::PVAxisComputationDlg::get_plugin()
{
	_cur_plugin->set_args(_plugin_args);
	return _cur_plugin;
}
