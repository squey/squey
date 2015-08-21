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
#include <pvkernel/core/PVProgressBox.h>

#include <QMessageBox>
#include <QPushButton>

enum EQueryType {
	QUERY_BUILDER = 0,
	JSON,
	SQL,

	COUNT
};
static const char* query_types[] = { "Query Builder", "JSON", "SQL" };

PVRush::PVElasticsearchParamsWidget::PVElasticsearchParamsWidget(PVInputTypeElasticsearch const* in_t, PVRush::hash_formats const& /*formats*/, QWidget* parent):
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
	setWindowTitle(tr("Import from Elasticsearch..."));
	setWindowIcon(in_t->icon());
	
	// Presets widget
	_presets_widget = new PVWidgets::PVPresetsWidget(tr("Saved settings"));
	_presets_widget->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Maximum);
	_presets_widget->setFixedHeight(250);
	presets_layout->addWidget(_presets_widget);
	populate_presets();

	// Set connections
	connect(_presets_widget, SIGNAL(btn_load_clicked_Signal(const QString&)), this, SLOT(preset_load_slot(const QString&)));
	connect(_presets_widget, SIGNAL(btn_new_clicked_Signal(const QString&)), this, SLOT(preset_new_slot(const QString&)));
	connect(_presets_widget, SIGNAL(btn_save_clicked_Signal(const QString&)), this, SLOT(preset_save_slot(const QString&)));
	connect(_presets_widget, SIGNAL(btn_remove_clicked_Signal(const QString&)), this, SLOT(preset_remove_slot(const QString&)));
	connect(_btn_browse_existing_format, SIGNAL(clicked()), this, SLOT(browse_existing_format_slot()));
	connect(_btn_edit_new, SIGNAL(clicked()), this, SLOT(edit_new_format_slot()));
	connect(_radio_use_existing, SIGNAL(toggled(bool)), this, SLOT(use_existing_format_toggle_slot(bool)));
	connect(_btn_refresh, SIGNAL(clicked()), this, SLOT(refresh_indexes()));
	::connect(_auth_enabled_cb, SIGNAL(stateChanged(int)), [&]{_auth_grp->setEnabled(_auth_enabled_cb->isChecked());});
	connect(_count_btn, SIGNAL(clicked()), this, SLOT(request_count()));
	connect(_query_type_cb, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(query_type_changed()));
	connect(_combo_index, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(index_changed(const QString&)));
	connect(_combo_index, SIGNAL(activated(int)), this, SLOT(index_changed_by_user_slot()));
	connect(_check_connection_push_button, SIGNAL(clicked()), this, SLOT(check_connection_slot()));
	connect(_export_pushbutton, SIGNAL(clicked()), this, SLOT(export_slot()));

	for (size_t i = 0 ; i < EQueryType::COUNT ; i++) {
		_query_type_cb->addItem(query_types[i]);
	}

	// Set SQL field columns
	_table_fields->setColumnCount(3);
	_table_fields->setHorizontalHeaderLabels(QStringList() << "Field name" << "SQL type" << "Picviz type");

	enable_used_format(true);

	// Hide "format" tab for the moment
	tabWidget->removeTab(2);

	// Get the last preset ID loaded
	_last_load_preset = -1;
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

QString PVRush::PVElasticsearchParamsWidget::get_sql_query_prefix() const
{
	return QString("SELECT * FROM %1 WHERE ").arg(get_infos().get_index());
}

void PVRush::PVElasticsearchParamsWidget::index_changed(const QString& index)
{
	buttonBox->buttons()[0]->setEnabled(index != "");

	QString query_type = _query_type_cb->currentText();

	if (query_type == query_types[EQueryType::SQL]) {
		_gb_query->setTitle(get_sql_query_prefix() + " ...");
	}
}

void PVRush::PVElasticsearchParamsWidget::index_changed_by_user_slot()
{
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::QUERY_BUILDER) {
		PVRush::PVElasticsearchAPI es(get_infos());
		_querybuilder->set_filters(es.columns(get_query()));
	}
}

bool PVRush::PVElasticsearchParamsWidget::check_connection()
{
	PVRush::PVElasticsearchAPI es(get_infos());

	return es.check_connection();
}

void PVRush::PVElasticsearchParamsWidget::check_connection_slot()
{
	PVRush::PVElasticsearchAPI es(get_infos());

	std::string error_msg;
	if (es.check_connection(&error_msg)) {
		QMessageBox::information(this, tr("Success"), tr("Connection successful"), QMessageBox::Ok);
		refresh_indexes();
	}
	else {
		QMessageBox::critical(this, tr("Failure"), tr("Connection error : %1").arg(error_msg.c_str()), QMessageBox::Ok);
	}
}

void PVRush::PVElasticsearchParamsWidget::query_type_changed()
{
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::SQL) {
		_txt_query->setPlainText("");
		_reference_label->setText(
			"<a href=\"https://github.com/NLPchina/elasticsearch-sql/\">"
			"<span style=\" text-decoration: underline; color:#0000ff;\">Elasticsearch SQL plugin"
		);
		PVRush::PVElasticsearchAPI es(get_infos());
		if (es.is_sql_available()) {
			_gb_query->setTitle(get_sql_query_prefix() + " ...");
			_txt_query->setEnabled(true);
		}
		else {
			_txt_query->setPlainText("Please, install the SQL plugin to your Elasticsearch instance to support this feature.");
			_txt_query->setEnabled(false);
			//buttonBox->buttons()[0]->setEnabled(false);
		}
		_querybuilder->setVisible(false);
		_txt_query->setVisible(true);
	}
	else if (query_type == EQueryType::QUERY_BUILDER) {
		_gb_query->setTitle("Query");
		_reference_label->setText("");
		_querybuilder->reset();
		_querybuilder->setVisible(true);
		_txt_query->setVisible(false);
	}
	else { // EQueryType::JSON
		_txt_query->setPlainText("\"query\" : { \"match_all\" : { } }");
		_gb_query->setTitle("Query");
		_reference_label->setText(
			"<a href=\"https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-filters.html\">"
			"<span style=\" text-decoration: underline; color:#0000ff;\">Elasticsearch Filters reference"
		);
		_querybuilder->setVisible(false);
		_txt_query->setVisible(true);
		_txt_query->setEnabled(true);
	}
}

void PVRush::PVElasticsearchParamsWidget::preset_new_slot(const QString& name)
{
	QString query = get_elasticsearch_query();
	QString query_type = get_query_type();

	// Set the new presets
	// ignore returned value
	PVElasticsearchPresets::get().add(name, get_infos(), query, query_type);
}

void PVRush::PVElasticsearchParamsWidget::preset_load_slot(const QString& /*preset*/)
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

	refresh_indexes();
	_query_type_cb->setCurrentIndex(_query_type_cb->findText(query_type));
	_combo_index->setCurrentIndex(_combo_index->findText(infos.get_index()));

	set_query(query);
	set_query_type(query_type);
	_last_load_preset = id;

	_presets_widget->select_preset(id-1);
}

void PVRush::PVElasticsearchParamsWidget::preset_save_slot(const QString& /*preset*/)
{
	PVElasticsearchPresets::id_t id = get_current_preset_id();
	QString query = get_serialized_query();
	QString query_type = get_query_type();

	PVElasticsearchPresets::get().set(id, get_infos(), query, query_type);
}

void PVRush::PVElasticsearchParamsWidget::preset_remove_slot(const QString& /*preset*/)
{
	PVElasticsearchPresets::id_t id = get_current_preset_id();
	PVElasticsearchPresets::get().rm(id);
}

void PVRush::PVElasticsearchParamsWidget::export_slot()
{
	std::string error;
	const PVRush::PVElasticsearchQuery& query = get_query(&error);

	if (error.empty()) {
		QString csv_filename = QFileDialog::getSaveFileName(
			this,
			"Export to...",
			"",
			QString("CSV File (*.csv);;All files (*.*)")
		);

		if (csv_filename.isEmpty() == false) {

			QFile f(csv_filename);
			if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {

				PVRush::PVElasticsearchAPI es(get_infos());

				QTextStream output_stream(&f);
				size_t count = 0;
				bool query_end = false;

				PVCore::PVProgressBox pbox("Exporting request result...");
				PVCore::PVProgressBox::progress([&]() {

					PVElasticsearchAPI::rows_chunk_t rows_array;
					query_end = es.extract(query, rows_array, &error);

					if (error.empty() == false) {
						return;
					}

					size_t max_count = es.scroll_count();
					pbox.getProgressBar()->setMaximum(max_count);

					while (query_end == false) {

						if (pbox.get_cancel_state() == PVCore::PVProgressBox::CANCEL ||
							pbox.get_cancel_state() == PVCore::PVProgressBox::CANCEL2) {
							break;
						}

						for (const PVElasticsearchAPI::rows_t& rows : rows_array) {
							for (const std::string& row : rows) {
								output_stream << row.c_str() << endl;
							}
							count += rows.size();
						}

						pbox.getProgressBar()->setValue(count);
					}
				}, &pbox);
			}
		}
	}

	if (error.empty() == false) {
		QMessageBox::critical((QWidget*) QObject::parent(), tr("Request failed"), tr("Request failed with the following error:\n\n%1").arg(QString(error.c_str())));
	}
}

PVRush::PVElasticsearchInfos PVRush::PVElasticsearchParamsWidget::get_infos() const
{
	PVElasticsearchInfos infos;

	infos.set_host(_txt_host->text());
	infos.set_port(_port_sb->value());
	infos.set_index(_combo_index->currentText());
	if (_auth_enabled_cb->isChecked()) {
		infos.set_login(_login_txt->text());
		infos.set_password(_passwd_txt->text());
	}

	return infos;
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

QString PVRush::PVElasticsearchParamsWidget::get_query_type() const
{
	return _query_type_cb->currentText();
}

QString PVRush::PVElasticsearchParamsWidget::get_elasticsearch_query(std::string* error /* = nullptr */) const
{
	QString q = get_serialized_query();
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::JSON) {
		q = _txt_query->toPlainText();
	}
	else {
		PVRush::PVElasticsearchAPI es(get_infos());

		if (query_type == EQueryType::SQL) {
			q = es.sql_to_json(QString(get_sql_query_prefix() + q).toStdString(), error).c_str();
		}
		else if (query_type == EQueryType::QUERY_BUILDER) {
			//q = es.rules_to_json(_querybuilder->get_rules());
		}
	}

	while(q.endsWith('\n')) q.chop(1);

	return q;
}

QString PVRush::PVElasticsearchParamsWidget::get_serialized_query() const
{
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::QUERY_BUILDER) {
		return QString(_querybuilder->get_rules().c_str());
	}
	else {
		return _txt_query->toPlainText();
	}
}

PVRush::PVElasticsearchQuery PVRush::PVElasticsearchParamsWidget::get_query(std::string* error /* = nullptr */) const
{
	PVElasticsearchQuery query(get_infos(), get_elasticsearch_query(error), get_query_type());

	return query;
}

void PVRush::PVElasticsearchParamsWidget::set_query(QString const& query)
{
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::QUERY_BUILDER) {
		PVRush::PVElasticsearchAPI es(get_infos());
		_querybuilder->set_filters(es.columns(get_query()));
		_querybuilder->set_rules(query.toStdString());
	}
	else {
		_txt_query->setPlainText(query);
	}
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
	PVRush::PVElasticsearchAPI es(get_infos());

	std::string sql_error;
	std::string query_error;
	size_t request_count = 0;
	PVCore::PVProgressBox pbox("Executing count request...");
	PVCore::PVProgressBox::progress([&]() {
		const PVElasticsearchQuery& query = get_query(&sql_error);
		request_count = es.count(query, &query_error);
	}, &pbox);

	if (sql_error.empty() && query_error.empty()) {
		QMessageBox::information((QWidget*) QObject::parent(), tr("Request count"), tr("The request returned %L1 result(s)").arg(request_count));
	}
	else
	{
		std::string error = sql_error.empty() == false ? sql_error : query_error;
		QMessageBox::critical((QWidget*) QObject::parent(), tr("Request failed"), tr("Request failed with the following error:\n\n%1").arg(QString(error.c_str())));
	}
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

void PVRush::PVElasticsearchParamsWidget::browse_existing_format_slot()
{
	QFileDialog fd;
	QString format_path = fd.getOpenFileName(0, QString("Select and existing format"), "."); //open file chooser

	_txt_format->setText(format_path);
}

void PVRush::PVElasticsearchParamsWidget::edit_new_format_slot()
{
	_in_t->edit_format(_new_format_doc, this);
}

void PVRush::PVElasticsearchParamsWidget::use_existing_format_toggle_slot(bool toggle)
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
	PVRush::PVElasticsearchAPI es(get_infos());
	PVRush::PVElasticsearchAPI::indexes_t indexes = es.indexes();

	QString old_index = _combo_index->currentText();

	_combo_index->clear();
	for (const std::string& index : indexes) {
		_combo_index->addItem(QString(index.c_str()));
	}

	_combo_index->setCurrentIndex(_combo_index->findText(old_index));
}
