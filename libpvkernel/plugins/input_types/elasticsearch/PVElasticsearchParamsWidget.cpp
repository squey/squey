/**
 * \file PVElasticsearchParamsWidget.cpp
 *
 * Copyright (C) Picviz Labs 2015
 */

#include "PVElasticsearchParamsWidget.h"
#include "PVInputTypeElasticsearch.h"
#include "../../common/elasticsearch/PVElasticsearchAPI.h"
#include "../../common/elasticsearch/PVElasticsearchInfos.h"
#include "../../common/elasticsearch/PVElasticsearchQuery.h"

#include <pvkernel/core/lambda_connect.h>

#include <QMessageBox>
#include <QPushButton>

PVRush::PVElasticsearchParamsWidget::PVElasticsearchParamsWidget(PVInputTypeElasticsearch const* in_t, PVRush::hash_formats const& /*formats*/, QWidget* parent):
	QDialog(parent),
	_settings(QSettings::UserScope, PICVIZ_ORGANISATION, PICVIZ_APPLICATIONNAME),
	_in_t(in_t)
{
	// Create the UI
	setupUi(this);

	// Set the dialog title
	setWindowTitle(tr("Import from Elasticsearch..."));
	
	// Presets widget
	_presets_widget = new PVWidgets::PVPresetsWidget(tr("Saved settings"));
	presets_layout->addWidget(_presets_widget);
	populate_presets();

	// Set connections
	connect(_presets_widget, SIGNAL(btn_load_clicked_Signal(const QString&)), this, SLOT(preset_load_Slot(const QString&)));
	connect(_presets_widget, SIGNAL(btn_new_clicked_Signal(const QString&)), this, SLOT(preset_new_Slot(const QString&)));
	connect(_presets_widget, SIGNAL(btn_save_clicked_Signal(const QString&)), this, SLOT(preset_save_Slot(const QString&)));
	connect(_presets_widget, SIGNAL(btn_remove_clicked_Signal(const QString&)), this, SLOT(preset_remove_Slot(const QString&)));
	connect(_btn_browse_existing_format, SIGNAL(clicked()), this, SLOT(browse_existing_format_Slot()));
	connect(_btn_edit_new, SIGNAL(clicked()), this, SLOT(edit_new_format_Slot()));
	connect(_radio_use_existing, SIGNAL(toggled(bool)), this, SLOT(use_existing_format_toggle_Slot(bool)));
	connect(_btn_refresh, SIGNAL(clicked()), this, SLOT(refresh_indexes()));
	::connect(_auth_enabled_cb, SIGNAL(stateChanged(int)), [&]{_auth_grp->setEnabled(_auth_enabled_cb->isChecked());});
	connect(_count_btn, SIGNAL(clicked()), this, SLOT(request_count()));
	connect(_query_type_cb, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(query_type_changed()));
	connect(_combo_index, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(index_changed(const QString&)));
	connect(_check_connection_push_button, SIGNAL(clicked()), this, SLOT(check_connection()));
	
	_last_load_preset = -1;

	// Get the last preset ID loaded
	_settings.beginGroup(PV_SETTINGS_INPUT_ELASTICSEARCH "ui");
	if (_settings.contains("last_preset")) {
		PVRush::PVElasticsearchPresets::id_t id = _settings.value("last_preset").toUInt();
		load_preset(id);
	}
	else {
		// Load the first preset if any
		if (_presets_widget->get_preset_count() > 0) {
			load_preset(_presets_widget->get_preset_data(0).toUInt());
		}
	}

	PVElasticsearchQuery query;
	get_query(query);

	_query_type_cb->addItem("JSON");
	_query_type_cb->addItem("SQL");

	// Set SQL field columns
	_table_fields->setColumnCount(3);
	_table_fields->setHorizontalHeaderLabels(QStringList() << "Field name" << "SQL type" << "Picviz type");

	enable_used_format(true);

	// Hide "format" tab for the moment
	tabWidget->removeTab(2);
}

PVRush::PVElasticsearchParamsWidget::~PVElasticsearchParamsWidget()
{
	if (_last_load_preset != -1) {
		_settings.setValue("last_preset", (PVElasticsearchPresets::id_t) _last_load_preset);
	}
}

void PVRush::PVElasticsearchParamsWidget::populate_presets()
{
	_presets_widget->clear_presets();

	// List presets
	PVElasticsearchPresets::list_id_names_t l = PVElasticsearchPresets::get().list_id_names();
	PVElasticsearchPresets::list_id_names_t::const_iterator it;
	for (it = l.begin(); it != l.end(); it++) {
		_presets_widget->add_preset(it->second,  it->first);
	}
}

PVRush::PVElasticsearchPresets::id_t PVRush::PVElasticsearchParamsWidget::get_current_preset_id()
{
	// This assume that an existing preset has been selected !
	//assert(!_presets_widget->is_preset_txt_new());
	return _presets_widget->get_preset_data().toUInt();
}

QString PVRush::PVElasticsearchParamsWidget::get_sql_query_prefix()
{
	PVElasticsearchInfos infos;
	get_infos(infos);

	return QString("SELECT * FROM %1 WHERE ").arg(infos.get_index());
}

void PVRush::PVElasticsearchParamsWidget::index_changed(const QString& index)
{
	buttonBox->buttons()[0]->setEnabled(index != "");
	refresh_query_groupbox();
}

void PVRush::PVElasticsearchParamsWidget::check_connection()
{
	PVElasticsearchInfos infos;
	get_infos(infos);
	PVRush::PVElasticsearchAPI es(infos);

	std::string error_msg;
	if (es.check_connection(&error_msg)) {
		QMessageBox::information(this, tr("Success"), tr("Connection successful"), QMessageBox::Ok);
	}
	else {
		QMessageBox::critical(this, tr("Failure"), tr("Connection error : %1").arg(error_msg.c_str()), QMessageBox::Ok);
	}
}

void PVRush::PVElasticsearchParamsWidget::refresh_query_groupbox()
{
	QString query_type = _query_type_cb->currentText();

	if (query_type == "SQL") {

		PVElasticsearchInfos infos;
		get_infos(infos);
		PVRush::PVElasticsearchAPI es(infos);

		if (es.is_sql_available()) {
			_gb_query->setTitle(get_sql_query_prefix() + " ...");
			_reference_label->setText("<a href=\"https://github.com/NLPchina/elasticsearch-sql/\"><span style=\" text-decoration: underline; color:#0000ff;\">Elasticsearch SQL plugin");
		}
		else {

		}
	}
	else {
		_gb_query->setTitle("Query");
		_reference_label->setText("<a href=\"https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-filters.html\"><span style=\" text-decoration: underline; color:#0000ff;\">Elasticsearch Filters reference");
	}
}

void PVRush::PVElasticsearchParamsWidget::query_type_changed()
{
	refresh_query_groupbox();
}

void PVRush::PVElasticsearchParamsWidget::preset_new_Slot(const QString& name)
{
	PVElasticsearchInfos new_infos;
	get_infos(new_infos);
	QString query = get_query();
	QString query_type = get_query_type();

	// Set the new presets
	// ignore returned value
	PVElasticsearchPresets::get().add(name, new_infos, query, query_type);
}

void PVRush::PVElasticsearchParamsWidget::preset_load_Slot(const QString& /*preset*/)
{
	PVElasticsearchPresets::id_t id = get_current_preset_id();
	load_preset(id);
}

void PVRush::PVElasticsearchParamsWidget::load_preset(PVElasticsearchPresets::id_t id)
{
	PVElasticsearchInfos infos;
	QString query;
	QString query_type;
	bool ret = PVElasticsearchPresets::get().get(id, infos, query, query_type);
	if (!ret) {
		// Maybe the user modified the settings by hand...
		QMessageBox msg(QMessageBox::Critical, tr("Error while loading preset..."), tr("Preset %1 could not be loaded. Maybe it has been modified and/or deleted by another application. The list of available presets will be refreshed.").arg(_presets_widget->get_current_preset_name()), QMessageBox::Ok);
		msg.exec();
		populate_presets();
		return;
	}

	if (!set_infos(infos)) {
		QMessageBox msg(QMessageBox::Warning, tr("Error while loading preset.."), tr("Error while loading preset.."), QMessageBox::Ok);
		msg.exec();
		return;
	}

	set_query(query);
	set_query_type(query_type);
	_last_load_preset = id;

	refresh_indexes();
	_combo_index->setCurrentIndex(_combo_index->findText(infos.get_index()));

	_query_type_cb->setCurrentIndex(_query_type_cb->findText(query_type));
}

void PVRush::PVElasticsearchParamsWidget::preset_save_Slot(const QString& /*preset*/)
{
	PVElasticsearchPresets::id_t id = get_current_preset_id();
	QString query = _txt_query->toPlainText();
	QString query_type = get_query_type();
	
	PVElasticsearchInfos new_infos;
	get_infos(new_infos);

	PVElasticsearchPresets::get().set(id, new_infos, query, query_type);
}

void PVRush::PVElasticsearchParamsWidget::preset_remove_Slot(const QString& /*preset*/)
{
	PVElasticsearchPresets::id_t id = get_current_preset_id();
	PVElasticsearchPresets::get().rm(id);
}

void PVRush::PVElasticsearchParamsWidget::get_infos(PVElasticsearchInfos& infos)
{
	infos.set_host(_txt_host->text());
	infos.set_port(_port_sb->value());
	infos.set_index(_combo_index->currentText());
	if (_auth_enabled_cb->isChecked()) {
		infos.set_login(_login_txt->text());
		infos.set_password(_passwd_txt->text());
	}
}

bool PVRush::PVElasticsearchParamsWidget::set_infos(PVElasticsearchInfos const& infos)
{
	_txt_host->setText(infos.get_host());
	_port_sb->setValue(infos.get_port());
	_combo_index->setCurrentIndex(_combo_index->findText(infos.get_index()));

	_auth_enabled_cb->setChecked(infos.get_login().isEmpty() == false);
	_login_txt->setText(infos.get_login());
	_passwd_txt->setText(infos.get_password());

	return true;
}

QString PVRush::PVElasticsearchParamsWidget::get_query_type()
{
	return _query_type_cb->currentText();
}

QString PVRush::PVElasticsearchParamsWidget::get_query()
{
	QString q = _txt_query->toPlainText();

	if (_query_type_cb->currentText() == "SQL") {
		PVElasticsearchInfos infos;
		get_infos(infos);
		PVRush::PVElasticsearchAPI es(infos);

		q = es.sql_to_json(QString(get_sql_query_prefix() + q).toStdString()).c_str();
	}

	while(q.endsWith('\n')) q.chop(1);

	return q;
}

void PVRush::PVElasticsearchParamsWidget::get_query(PVElasticsearchQuery& query)
{
	PVElasticsearchInfos& infos(query.get_infos());
	get_infos(infos);

	query.set_query(get_query());
	query.set_query_type(get_query_type());
}

void PVRush::PVElasticsearchParamsWidget::set_query(QString const& query)
{
	_txt_query->setPlainText(query);
}

void PVRush::PVElasticsearchParamsWidget::set_query_type(QString const& query_type)
{
	_query_type_cb->setCurrentIndex(_query_type_cb->findText(query_type));
}

void PVRush::PVElasticsearchParamsWidget::show_def_params()
{
	show_layout_children(_layout_host, true);
}

void PVRush::PVElasticsearchParamsWidget::request_count()
{
	PVRush::PVElasticsearchQuery query;
	get_query(query);

	PVElasticsearchInfos infos;
	get_infos(infos);

	PVRush::PVElasticsearchAPI es(infos);
	size_t request_count = es.count(query);

	QMessageBox::information((QWidget*) QObject::parent(), tr("Request count"), tr("The request returned %L1 result(s)").arg(request_count));
}

void PVRush::PVElasticsearchParamsWidget::show_layout_children(const QLayout* layout, bool show)
{
	QLayoutItem *item = 0;
	QWidget *widget = 0;

	for(int i = 0; i < layout->count(); ++i)
	{
		item = layout->itemAt(i);
		if (item) {
			widget = item->widget();
			if (widget) {
				widget->setVisible(show);
			}
		}
	}
}

void PVRush::PVElasticsearchParamsWidget::browse_existing_format_Slot()
{
	QFileDialog fd;
	QString format_path = fd.getOpenFileName(0, QString("Select and existing format"), "."); //open file chooser

	_txt_format->setText(format_path);
}

void PVRush::PVElasticsearchParamsWidget::edit_new_format_Slot()
{
	_in_t->edit_format(_new_format_doc, this);
}

void PVRush::PVElasticsearchParamsWidget::use_existing_format_toggle_Slot(bool toggle)
{
	enable_used_format(toggle);
}

void PVRush::PVElasticsearchParamsWidget::enable_used_format(bool is_existing)
{
	_btn_browse_existing_format->setEnabled(is_existing);

	_btn_update_fields->setEnabled(!is_existing);
	_btn_edit_new->setEnabled(!is_existing);
	_table_fields->setEnabled(!is_existing);
	_btn_saveas->setEnabled(!is_existing);
}

QString PVRush::PVElasticsearchParamsWidget::get_existing_format()
{
	return PICVIZ_BROWSE_FORMAT_STR;
}

void PVRush::PVElasticsearchParamsWidget::refresh_indexes()
{
	PVElasticsearchInfos infos;
	get_infos(infos);

	PVRush::PVElasticsearchAPI es(infos);
	PVRush::PVElasticsearchAPI::indexes_t indexes = es.indexes();

	QString old_index = _combo_index->currentText();

	_combo_index->clear();
	for (const std::string& index : indexes) {
		_combo_index->addItem(QString(index.c_str()));
	}

	_combo_index->setCurrentIndex(_combo_index->findText(old_index));
}
