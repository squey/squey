/**
 * @file
 *
 * @copyright (C) Picviz Labs 2015
 */

#include "PVParamsWidget.h"

#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/widgets/PVQueryBuilder.h>
#include <pvkernel/core/lambda_connect.h>

/******************************************************************************
 *
 * PVRush::PVParamsWidgetBase::PVParamsWidgetBase
 *
 *****************************************************************************/
PVRush::PVParamsWidgetBase::PVParamsWidgetBase(PVInputType const* in_t, PVRush::hash_formats const& /*formats*/, QWidget* parent):
	QDialog(parent),
	_settings(QSettings::UserScope, PICVIZ_ORGANISATION, PICVIZ_APPLICATIONNAME),
	_in_t(in_t)
{
	// Create the UI
	setupUi(this);

	// Create the Query Builder
	_querybuilder = new PVWidgets::PVQueryBuilder(this);
	_querybuilder_layout->addWidget(_querybuilder);

	// Set the dialog title
	setWindowTitle("Import from " + in_t->menu_input_name());
	setWindowIcon(in_t->icon());
	tabWidget->setTabText(0, in_t->human_name_serialize() + " connection");

	// Presets widget
	_presets_widget = new PVWidgets::PVPresetsWidget(tr("Saved settings"));
	_presets_widget->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Maximum);
	_presets_widget->setFixedHeight(250);
	presets_layout->addWidget(_presets_widget);

	// Set connections
	connect(_presets_widget, SIGNAL(btn_load_clicked_Signal(const QString&)), this, SLOT(preset_load_slot()));
	connect(_presets_widget, SIGNAL(btn_new_clicked_Signal(const QString&)), this, SLOT(preset_new_slot(const QString&)));
	connect(_presets_widget, SIGNAL(btn_save_clicked_Signal(const QString&)), this, SLOT(preset_save_slot()));
	connect(_presets_widget, SIGNAL(btn_remove_clicked_Signal(const QString&)), this, SLOT(preset_remove_slot()));
	::connect(_auth_enabled_cb, SIGNAL(stateChanged(int)), [&]{_auth_grp->setEnabled(_auth_enabled_cb->isChecked());});
	connect(_count_btn, SIGNAL(clicked()), this, SLOT(query_result_count_slot()));
	connect(_query_type_cb, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(query_type_changed_slot()));
	connect(_check_connection_push_button, SIGNAL(clicked()), this, SLOT(check_connection_slot()));
	connect(_export_pushbutton, SIGNAL(clicked()), this, SLOT(export_slot()));
	connect(this, SIGNAL(load_preset_deferred(unsigned int)), this, SLOT(load_preset(unsigned int)), Qt::QueuedConnection);

	// Hide "format" tab for the moment
	tabWidget->removeTab(2);
}

QString PVRush::PVParamsWidgetBase::get_query_type() const
{
	return _query_type_cb->currentText();
}

void PVRush::PVParamsWidgetBase::set_query_type(QString const& query_type)
{
	_query_type_cb->setCurrentIndex(_query_type_cb->findText(query_type));
}

void PVRush::PVParamsWidgetBase::check_connection_slot()
{
	std::string error;

	if (check_connection(&error)) {
		QMessageBox::information(
			this,
			tr("Success"),
			tr("Connection successful"),
			QMessageBox::Ok
		);
	}
	else {
		QMessageBox::critical(
			this,
			tr("Failure"),
			tr("Connection error : %1").arg(error.c_str()), QMessageBox::Ok
		);
	}
}

void PVRush::PVParamsWidgetBase::query_result_count_slot()
{
	std::string error;

	size_t count = 0;
	PVCore::PVProgressBox pbox("Executing count request...");
	PVCore::PVProgressBox::progress([&]() {
		count = query_result_count(&error);
	}, &pbox);

	if (error.empty()) {
		QMessageBox::information(
			(QWidget*) QObject::parent(),
			tr("Request count"),
			tr("The request returned %L1 result(s)").arg(count));
	}
	else
	{
		QMessageBox::critical(
			(QWidget*) QObject::parent(),
			tr("Request failed"),
			tr("Request failed with the following error:\n\n%1").arg(QString(error.c_str()))
		);
	}
}
