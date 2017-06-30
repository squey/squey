/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include "PVParamsWidget.h"

#include <pvbase/general.h>

#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/widgets/PVQueryBuilder.h>

/******************************************************************************
 *
 * PVRush::PVParamsWidgetBase::PVParamsWidgetBase
 *
 *****************************************************************************/
PVRush::PVParamsWidgetBase::PVParamsWidgetBase(PVInputType const* in_t,
                                               PVRush::hash_formats const& /*formats*/,
                                               QWidget* parent)
    : QDialog(parent)
    , _settings(QSettings::UserScope, INENDI_ORGANISATION, INENDI_APPLICATIONNAME)
    , _in_t(in_t)
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
	_presets_widget->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
	_presets_widget->setFixedHeight(250);
	presets_layout->addWidget(_presets_widget);

	_open_format_button->setIcon(QIcon(":/document-open.png"));
	_edit_existing_format_button->setIcon(QIcon(":/edit-copy.png"));
	_edit_custom_format_button->setIcon(QIcon(":/edit-copy.png"));

	// Set connections
	connect(_presets_widget, &PVWidgets::PVPresetsWidget::btn_load_clicked_Signal, this,
	        &PVParamsWidgetBase::preset_load_slot);
	connect(_presets_widget, &PVWidgets::PVPresetsWidget::btn_new_clicked_Signal, this,
	        &PVParamsWidgetBase::preset_new_slot);
	connect(_presets_widget, &PVWidgets::PVPresetsWidget::btn_save_clicked_Signal, this,
	        &PVParamsWidgetBase::preset_save_slot);
	connect(_auth_enabled_cb, &QCheckBox::stateChanged,
	        [&] { _auth_grp->setEnabled(_auth_enabled_cb->isChecked()); });
	connect(_count_btn, &QPushButton::clicked, this, &PVParamsWidgetBase::query_result_count_slot);
	connect(_query_type_cb, SIGNAL(currentIndexChanged(const QString&)), this,
	        SLOT(query_type_changed_slot()));
	connect(_check_connection_push_button, &QPushButton::clicked, this,
	        &PVParamsWidgetBase::check_connection_slot);
	connect(_export_pushbutton, &QPushButton::clicked, this, &PVParamsWidgetBase::export_slot);
	connect(this, &PVParamsWidgetBase::load_preset_deferred, this, &PVParamsWidgetBase::load_preset,
	        Qt::QueuedConnection);
	connect(_open_format_button, &QPushButton::clicked, this, &PVParamsWidgetBase::load_format);
	connect(_edit_existing_format_button, &QPushButton::clicked, this,
	        &PVParamsWidgetBase::edit_existing_format);
	connect(_edit_custom_format_button, &QPushButton::clicked, this,
	        &PVParamsWidgetBase::edit_custom_format);
}

QString PVRush::PVParamsWidgetBase::get_query_type() const
{
	return _query_type_cb->currentText();
}

void PVRush::PVParamsWidgetBase::set_query_type(QString const& query_type)
{
	_query_type_cb->setCurrentIndex(_query_type_cb->findText(query_type));
}

void PVRush::PVParamsWidgetBase::load_format()
{
	QFileDialog file_dialog;
	file_dialog.setWindowTitle("Load format from...");
	file_dialog.setAcceptMode(QFileDialog::AcceptOpen);
	file_dialog.setOption(QFileDialog::DontUseNativeDialog, true);
	file_dialog.setNameFilters(QStringList{"Formats (*.format)", "All files (*.*)"});

	if (file_dialog.exec()) {
		const QString format_path = file_dialog.selectedFiles().at(0);
		if (not format_path.isEmpty()) {
			_format_path->setText(format_path);
		}
	}
}

void PVRush::PVParamsWidgetBase::edit_existing_format()
{
	_in_t->edit_format(_format_path->text(), this);
}

void PVRush::PVParamsWidgetBase::edit_custom_format()
{
	_in_t->edit_format(_custom_format, this);
}

/*****************************************************************************
 * PVRush::PVParamsWidgetBase::get_export_filters
 *****************************************************************************/

QString PVRush::PVParamsWidgetBase::get_export_filters()
{
	return "All files (*)";
}

void PVRush::PVParamsWidgetBase::check_connection_slot()
{
	std::string error;

	if (check_connection(&error)) {
		QMessageBox::information(this, tr("Success"), tr("Connection successful"), QMessageBox::Ok);
	} else {
		QMessageBox::critical(this, tr("Failure"), tr("Connection error : %1").arg(error.c_str()),
		                      QMessageBox::Ok);
	}
}

void PVRush::PVParamsWidgetBase::query_result_count_slot()
{
	std::string error;

	size_t count = 0;
	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& /*pbox*/) { count = query_result_count(&error); },
	    "Executing count request...", this);

	if (error.empty()) {
		QMessageBox::information(this, tr("Request count"),
		                         tr("The request returned %L1 result(s)").arg(count));
	} else {
		QMessageBox::critical(
		    this, tr("Request failed"),
		    tr("Request failed with the following error:\n\n%1").arg(QString(error.c_str())));
	}
}
