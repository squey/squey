#include "PVDatabaseParamsWidget.h"
#include "PVDBPreviewWidget.h"
#include "PVSQLTypeMap.h"
#include "PVInputTypeDatabase.h"

#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlRecord>
#include <QSqlField>
#include <QHash>
#include <QMessageBox>


// Hash table used to have 'human-readable' name of Qt's SQL dirvers
class HashDriversName
{
public:
	HashDriversName()
	{
		_hash["QMYSQL"] = "MySQL";
		_hash["QMYSQL3"] = "MySQL 3.x";
		_hash["QSQLITE"] = "SQLite";
		_hash["ODBC"] = "ODBC";

		_port["QMYSQL"] = 3306;
		_port["QMYSQL3"] = 3306;
		_port["QSQLITE"] = 0;
		_port["ODBC"] = 0;
	}
	QString name(QString const& driver)
	{ 
		if (_hash.contains(driver)) {
			return _hash[driver];
		}
		return driver;
	}

	uint16_t port(QString const& driver)
	{
		if (_port.contains(driver)) {
			return _port[driver];
		}
		return 0;
	}
private:
	QHash<QString,QString> _hash;
	QHash<QString,uint16_t> _port;
};
static HashDriversName g_drivers_name;

PVRush::PVDatabaseParamsWidget::PVDatabaseParamsWidget(PVInputTypeDatabase const* in_t, PVRush::hash_formats const& formats, QWidget* parent):
	QDialog(parent),
	_settings(QSettings::UserScope, PICVIZ_ORGANISATION, PICVIZ_APPLICATIONNAME),
	_in_t(in_t)
{
	// Create the UI
	setupUi(this);

	// Set the dialog title
	setWindowTitle(tr("Import from a database..."));

	// List the supported QtSql drivers
	QStringList drivers = QSqlDatabase::drivers();
	for (int i = 0; i < drivers.size(); i++) {
		QString d = drivers[i];
		_combo_type->addItem(g_drivers_name.name(d), d);
	}
	
	populate_presets();

	// Set connections
	connect(_btn_new, SIGNAL(clicked()), this, SLOT(preset_new_Slot()));
	connect(_btn_load, SIGNAL(clicked()), this, SLOT(preset_load_Slot()));
	connect(_btn_save, SIGNAL(clicked()), this, SLOT(preset_save_Slot()));
	connect(_btn_remove, SIGNAL(clicked()), this, SLOT(preset_remove_Slot()));
	connect(_combo_preset, SIGNAL(editTextChanged(const QString&)), this, SLOT(preset_text_changed_Slot(const QString&)));
	connect(_combo_type, SIGNAL(currentIndexChanged(int)), this, SLOT(sql_type_changed_Slot(int)));
	connect(_btn_query_preview, SIGNAL(clicked()), this, SLOT(query_preview_Slot()));
	connect(_btn_update_fields, SIGNAL(clicked()), this, SLOT(update_fields_Slot()));
	connect(_btn_edit_existing, SIGNAL(clicked()), this, SLOT(edit_existing_format_Slot()));
	connect(_btn_edit_new, SIGNAL(clicked()), this, SLOT(edit_new_format_Slot()));
	connect(_radio_use_existing, SIGNAL(toggled(bool)), this, SLOT(use_existing_format_toggle_Slot(bool)));
	
	_last_load_preset = -1;

	// Get the last preset ID loaded
	_settings.beginGroup(PV_SETTINGS_INPUT_DB "ui");
	if (_settings.contains("last_preset")) {
		PVRush::PVDBPresets::id_t id = _settings.value("last_preset").toUInt();
		load_preset(id);
	}
	else {
		// Load the first preset if any
		if (_combo_preset->count() > 0) {
			load_preset(_combo_preset->itemData(0).toUInt());
		}
		else {
			_combo_type->setCurrentIndex(0);
		}
	}
	// Get the latest number of rows used
	uint32_t nrows;
	nrows = _settings.value("preview_nrows", 10).toUInt();
	_txt_nrows->setText(QString::number(nrows));
	preset_text_changed_Slot(_combo_preset->lineEdit()->text());

	// Set the existing formats
	PVRush::hash_formats::const_iterator it;
	for (it = formats.begin(); it != formats.end(); it++) {
		_combo_formats->addItem(it.key(), it.value().get_full_path());
	}
	_btn_edit_existing->setEnabled(_combo_formats->count() > 0);

	// Set SQL field columns
	_table_fields->setColumnCount(3);
	_table_fields->setHorizontalHeaderLabels(QStringList() << "Field name" << "SQL type" << "Picviz type");

	enable_used_format(true);
}

PVRush::PVDatabaseParamsWidget::~PVDatabaseParamsWidget()
{
	if (_last_load_preset != -1) {
		_settings.setValue("last_preset", (PVDBPresets::id_t) _last_load_preset);
	}
	if (!_txt_nrows->text().isEmpty()) {
		_settings.setValue("preview_nrows", _txt_nrows->text().toUInt());
	}
}

void PVRush::PVDatabaseParamsWidget::populate_presets()
{
	_combo_preset->clear();

	// List presets
	PVDBPresets::list_id_names_t l = PVDBPresets::get().list_id_names();
	PVDBPresets::list_id_names_t::const_iterator it;
	for (it = l.begin(); it != l.end(); it++) {
		add_preset(it->second, it->first);
	}
}

void PVRush::PVDatabaseParamsWidget::add_preset(QString const& name, PVDBPresets::id_t id)
{
	_combo_preset->addItem(name, id);
}

PVRush::PVDBPresets::id_t PVRush::PVDatabaseParamsWidget::get_current_preset_id()
{
	// This assume that an existing preset has been selected !
	assert(!is_preset_txt_new());
	return _combo_preset->itemData(_combo_preset->currentIndex()).toUInt();
}

void PVRush::PVDatabaseParamsWidget::preset_new_Slot()
{
	PVDBInfos new_infos;
	get_dbinfos(new_infos);
	QString query = get_query();
	QString name = _combo_preset->lineEdit()->text();

	// Set the new presets
	PVDBPresets::id_t id = PVDBPresets::get().add(name, new_infos, query);

	// And refresh everything
	add_preset(name, id);
	preset_text_changed_Slot(name);
}

void PVRush::PVDatabaseParamsWidget::preset_load_Slot()
{
	PVDBPresets::id_t id = get_current_preset_id();
	load_preset(id);
}

void PVRush::PVDatabaseParamsWidget::load_preset(PVDBPresets::id_t id)
{
	PVDBInfos infos;
	QString query;
	bool ret = PVDBPresets::get().get(id, infos, query);
	if (!ret) {
		// Maybe the user modified the settings by hand...
		QMessageBox msg(QMessageBox::Critical, tr("Error while loading preset..."), tr("Preset %1 could not be loaded. Maybe it has been modified and/or deleted by another application. The list of available presets will be refreshed.").arg(_combo_preset->currentText()), QMessageBox::Ok);
		msg.exec();
		populate_presets();
		return;
	}

	if (!set_dbinfos(infos)) {
		QMessageBox msg(QMessageBox::Warning, tr("Error while loading preset.."), tr("Database driver %1 isn't supported !").arg(g_drivers_name.name(infos.get_type())), QMessageBox::Ok);
		msg.exec();
		return;
	}

	set_query(query);
	_last_load_preset = id;
}

void PVRush::PVDatabaseParamsWidget::preset_save_Slot()
{
	PVDBPresets::id_t id = get_current_preset_id();
	QString query = get_query();
	
	PVDBInfos new_infos;
	get_dbinfos(new_infos);

	PVDBPresets::get().set(id, new_infos, query);
}

void PVRush::PVDatabaseParamsWidget::preset_remove_Slot()
{
	PVDBPresets::id_t id = get_current_preset_id();
	PVDBPresets::get().rm(id);
	populate_presets();
}

bool PVRush::PVDatabaseParamsWidget::is_preset_txt_new()
{
	return _combo_preset->findText(_combo_preset->lineEdit()->text(), Qt::MatchFixedString) == -1;
}

void PVRush::PVDatabaseParamsWidget::preset_text_changed_Slot(const QString& text)
{
	if (text.isEmpty()) {
		_btn_new->setEnabled(false);
		_btn_save->setEnabled(false);
		_btn_load->setEnabled(false);
		_btn_remove->setEnabled(false);
		return;
	}

	bool preset_new = is_preset_txt_new();
	_btn_new->setEnabled(preset_new);
	_btn_save->setEnabled(!preset_new);
	_btn_load->setEnabled(!preset_new);
	_btn_remove->setEnabled(!preset_new);
}

void PVRush::PVDatabaseParamsWidget::get_dbinfos(PVDBInfos& infos)
{
	infos.set_type(get_current_driver());
	infos.set_host(_txt_host->text());
	infos.set_username(_txt_user->text());
	infos.set_password(_txt_user->text());
	infos.set_port(_txt_port->text().toUInt());
	infos.set_dbname(_txt_dbname->text());
}

QString PVRush::PVDatabaseParamsWidget::get_current_driver()
{
	return _combo_type->itemData(_combo_type->currentIndex()).toString();
}

QString PVRush::PVDatabaseParamsWidget::get_query()
{
	return _txt_query->toPlainText();
}

bool PVRush::PVDatabaseParamsWidget::set_dbinfos(PVDBInfos const& infos)
{
	if (!select_type(infos.get_type())) {
		return false;
	}
	_txt_host->setText(infos.get_host());
	_txt_user->setText(infos.get_username());
	_txt_pwd->setText(infos.get_password());
	_txt_port->setText(QString::number(infos.get_port()));
	_txt_dbname->setText(infos.get_dbname());
	return true;
}

void PVRush::PVDatabaseParamsWidget::set_query(QString const& query)
{
	_txt_query->setPlainText(query);
}

bool PVRush::PVDatabaseParamsWidget::select_type(QString const& qt_type)
{
	int index = _combo_type->findData(qt_type);
	if (index == -1) {
		return false;
	}

	_combo_type->setCurrentIndex(index);
	return true;
}


void PVRush::PVDatabaseParamsWidget::sql_type_changed_Slot(int idx)
{
	_txt_port->setText(QString::number(g_drivers_name.port(_combo_type->itemData(idx).toString())));
}

void PVRush::PVDatabaseParamsWidget::query_preview_Slot()
{
	// Get current infos
	PVDBInfos new_infos;
	get_dbinfos(new_infos);

	PVDBPreviewWidget* dlg = new PVDBPreviewWidget(new_infos, get_query(), _txt_nrows->text().toUInt(), this);

	if (!dlg->init()) {
		return;
	}
	dlg->show();
	dlg->preview();
}

void PVRush::PVDatabaseParamsWidget::update_fields_Slot()
{
	//_table_fields->clear();

	PVDBInfos new_infos;
	get_dbinfos(new_infos);

	// Get a valid QSqlRecord
	PVDBServ_p srv(new PVDBServ(new_infos));
	PVDBQuery qr(srv, get_query());
	if (!qr.connect_serv()) {
		return;
	}
	QSqlQuery query = qr.to_query(0,1);
	query.exec();
	QSqlRecord record = query.record();

	// Create an XML dom representation that corresponds to a format
	// that goes w/ these fields
	_new_format_doc.clear();
	PVXmlTreeNodeDom* new_format_root = PVRush::PVXmlTreeNodeDom::new_format(_new_format_doc);

	PVSQLTypeMap_p type_map = PVSQLTypeMap::get_map(get_current_driver());
	// Go through that record
	int nfields = record.count();
	_table_fields->setRowCount(nfields);
	for (int i = 0; i < nfields; i++) {
		QSqlField field = record.field(i);
		QString name = field.name();
		_table_fields->setItem(i, 0, new QTableWidgetItem(name));
		// typeID isn't documented ! (well its name is in the detailed description of QSqlField, but that's it !)
		int type_id = field.typeID();
		_table_fields->setItem(i, 1, new QTableWidgetItem(type_map->map(type_id)));

		QString pv_type = type_map->map_picviz(type_id);
		_table_fields->setItem(i, 2, new QTableWidgetItem(pv_type));
		new_format_root->addOneField(name, pv_type);
	}
}

void PVRush::PVDatabaseParamsWidget::edit_existing_format_Slot()
{
	QString path = _combo_formats->itemData(_combo_formats->currentIndex()).toString();
	_in_t->edit_format(path, this);
}

void PVRush::PVDatabaseParamsWidget::edit_new_format_Slot()
{
	_in_t->edit_format(_new_format_doc, this);
}

void PVRush::PVDatabaseParamsWidget::use_existing_format_toggle_Slot(bool toggle)
{
	enable_used_format(toggle);
}

void PVRush::PVDatabaseParamsWidget::enable_used_format(bool is_existing)
{
	_combo_formats->setEnabled(is_existing);
	_btn_edit_existing->setEnabled(is_existing);

	_btn_update_fields->setEnabled(!is_existing);
	_btn_edit_new->setEnabled(!is_existing);
	_table_fields->setEnabled(!is_existing);
	_btn_saveas->setEnabled(!is_existing);
}

