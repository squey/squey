#include "PVArcsightParamsWidget.h"
#include "PVInputTypeArcsight.h"

#include <pvkernel/core/PVProgressBox.h>

#include <QMessageBox>
#include <QPushButton>
#include <QFileDialog>

PVRush::PVArcsightParamsWidget::PVArcsightParamsWidget(PVInputTypeArcsight const* in_t, PVRush::hash_formats const& formats, QWidget* parent):
	QDialog(parent),
	_in_t(in_t)
{
	// Create the UI
	setupUi(this);

	// Set the dialog title
	setWindowTitle(tr("Import from Arcsight..."));
	//connect(_btn_query_preview, SIGNAL(clicked()), this, SLOT(query_preview()));
	connect(_btn_update_fields, SIGNAL(clicked()), this, SLOT(update_fields_list()));
	connect(_radio_use_existing, SIGNAL(toggled(bool)), this, SLOT(use_existing_format_toggle(bool)));
	connect(_btn_edit_new, SIGNAL(clicked()), this, SLOT(edit_new_format()));
	
	// Set the existing formats
	PVRush::hash_formats::const_iterator it;
	for (it = formats.begin(); it != formats.end(); it++) {
		_combo_formats->addItem(it.key(), it.value().get_full_path());
	}
	_btn_edit_existing->setEnabled(_combo_formats->count() > 0);

	enable_used_format(true);
}

PVRush::PVArcsightParamsWidget::~PVArcsightParamsWidget()
{
}

void PVRush::PVArcsightParamsWidget::get_infos(PVArcsightInfos& infos)
{
	infos.set_host(_txt_host->text());
	infos.set_username(_txt_user->text());
	infos.set_password(_txt_pwd->text());
	infos.set_port(_txt_port->text().toUInt());
}

void PVRush::PVArcsightParamsWidget::get_query(PVArcsightQuery& query)
{
	set_list_fields_kept();

	PVArcsightInfos& infos(query.get_infos());
	get_infos(infos);
	query.set_query(get_query());
	query.get_fields_kept() = _fields_kept;
}

QString PVRush::PVArcsightParamsWidget::get_query()
{
	return _txt_query->toPlainText();
}

bool PVRush::PVArcsightParamsWidget::set_infos(PVArcsightInfos const& infos)
{
	_txt_host->setText(infos.get_host());
	_txt_user->setText(infos.get_username());
	_txt_pwd->setText(infos.get_password());
	_txt_port->setText(QString::number(infos.get_port()));
	return true;
}

void PVRush::PVArcsightParamsWidget::set_query(QString const& query)
{
	_txt_query->setPlainText(query);
}

arcsight::ArcsightLogger PVRush::PVArcsightParamsWidget::connect_to_arcsight()
{
	arcsight::ArcsightLogger arcsight;
	arcsight.login(qPrintable(_txt_host->text()), _txt_port->text().toUInt(),
	               _txt_user->text().toStdString(), _txt_pwd->text().toStdString());
	return arcsight;
}

void PVRush::PVArcsightParamsWidget::update_fields_list()
{
	_fields_name.clear();

	PVCore::PVProgressBox* pbox = new PVCore::PVProgressBox(tr("Updating fields..."), this);
	pbox->set_enable_cancel(false);
	QString err_msg;
	PVCore::PVProgressBox::progress([&] {
		try {
			arcsight::ArcsightLogger arcsight = connect_to_arcsight();
			arcsight::ArcsightLoggerSearch search = arcsight.search(qPrintable(get_query()), 0, (int64_t)(time(NULL))*1000);

			std::vector<std::string> headers;
			search.get_header(headers);
			_fields_name.reserve(headers.size());
			for (std::string const& s: headers) {
				_fields_name.push_back(QString::fromUtf8(s.c_str(), s.size()));
			}

			search.close();
			arcsight.logout();
		}
		catch (arcsight::Exception const& e) {
			err_msg = tr("Error while getting fields: \n") + e.what();
		}
	}, pbox);
	pbox->deleteLater();

	if (!err_msg.isEmpty()) {
		QMessageBox::critical(this, tr("Error while getting fields"), err_msg);
		_fields_name.clear();
	}

	update_fields_list_format();
}

void PVRush::PVArcsightParamsWidget::set_list_fields_kept()
{
	_fields_kept.clear();
	for (int i = 0; i < _list_fields->count(); i++) {
		if (_list_fields->item(i)->checkState() == Qt::Checked) {
			_fields_kept.insert(i);
		}
	}
}

void PVRush::PVArcsightParamsWidget::update_fields_list_format()
{
	_list_fields->clear();
	for (QString const& field_name: _fields_name) {
		QListWidgetItem* item = new QListWidgetItem(field_name);
		item->setCheckState(Qt::Checked);
		_list_fields->insertItem(_list_fields->count(), item);
	}

#if 0
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
#endif
}

void PVRush::PVArcsightParamsWidget::create_new_format()
{
	set_list_fields_kept();

	_new_format_doc.clear();
	PVXmlTreeNodeDom* new_format_root = PVRush::PVXmlTreeNodeDom::new_format(_new_format_doc);

	for (uint32_t i: _fields_kept) {
		new_format_root->addOneField(_fields_name[i], "enum");
	}
}

void PVRush::PVArcsightParamsWidget::edit_existing_format()
{
	QString path = _combo_formats->itemData(_combo_formats->currentIndex()).toString();
	_in_t->edit_format(path, this);
}

void PVRush::PVArcsightParamsWidget::edit_new_format()
{
	create_new_format();
	_in_t->edit_format(_new_format_doc, this);
}

void PVRush::PVArcsightParamsWidget::use_existing_format_toggle(bool toggle)
{
	enable_used_format(toggle);
}

void PVRush::PVArcsightParamsWidget::enable_used_format(bool is_existing)
{
	_combo_formats->setEnabled(is_existing);
	_btn_edit_existing->setEnabled(is_existing);

	_btn_update_fields->setEnabled(!is_existing);
	_btn_edit_new->setEnabled(!is_existing);
	_list_fields->setEnabled(!is_existing);
	_btn_saveas->setEnabled(!is_existing);
}

QString PVRush::PVArcsightParamsWidget::get_existing_format()
{
	return _combo_formats->currentText();
}
