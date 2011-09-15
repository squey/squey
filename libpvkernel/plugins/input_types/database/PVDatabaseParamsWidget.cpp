#include "PVDatabaseParamsWidget.h"
#include "PVDBPresets.h"

#include <QSqlDatabase>
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
	}
	QString name(QString const& driver) { return _hash[driver]; }
private:
	QHash<QString,QString> _hash;
};
static HashDriversName g_drivers_name;

PVRush::PVDatabaseParamsWidget::PVDatabaseParamsWidget(QWidget* parent):
	QDialog(parent)
{
	// Create the UI
	setupUi(this);

	// List the supported QtSql drivers
	QStringList drivers = QSqlDatabase::drivers();
	for (int i = 0; i < drivers.size(); i++) {
		QString d = drivers[i];
		_combo_type->addItem(g_drivers_name.name(d), d);
	}

	// List presets
	PVDBPresets::list_id_names_t l = PVDBPresets::get().list_id_names();
	PVDBPresets::list_id_names_t::const_iterator it;
	for (it = l.begin(); it != l.end(); it++) {
		//_combo_presets.addItem(it->second, it->first);
	}

	// Set connections
	connect(_btn_new, SIGNAL(clicked()), this, SLOT(preset_new_Slot()));
	connect(_btn_save, SIGNAL(clicked()), this, SLOT(preset_save_Slot()));
}

void PVRush::PVDatabaseParamsWidget::preset_new_Slot()
{
	QMessageBox::information(this, "hi", "hi");
}

#if 0
void PVRush::PVDatabaseParamsWidget::get_dbinfos()
{
}
#endif
