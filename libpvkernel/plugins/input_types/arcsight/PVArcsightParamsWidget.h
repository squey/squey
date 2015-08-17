#ifndef PVDATABASEPARAMSWIDGET_H
#define PVDATABASEPARAMSWIDGET_H

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <pvkernel/widgets/PVPresetsWidget.h>

#include <ui_arcsight_params.h>

#include "../../common/arcsight/PVArcsightQuery.h"

#include <arcsight/ArcsightLogger.h>

#include <QDomDocument>


namespace PVRush {

class PVInputTypeArcsight;

class PVArcsightParamsWidget: public QDialog, private Ui::ArcsightParams
{
	Q_OBJECT

public:
	PVArcsightParamsWidget(PVInputTypeArcsight const* in_t, PVRush::hash_formats const& formats, QWidget* parent);
	~PVArcsightParamsWidget();

public:
	QString get_query();
	bool is_format_custom() { return _radio_new_format->isChecked(); };
	QString get_existing_format();
	QDomDocument get_custom_format() { return _new_format_doc; };
	void get_query(PVArcsightQuery& query);
	void get_infos(PVArcsightInfos& infos);
	bool set_infos(PVArcsightInfos const& infos);

public slots:
	void preset_text_changed(const QString& text);
	void query_preview();
	void edit_existing_format();
	void edit_new_format();
	void use_existing_format_toggle(bool toggle);

protected:
	void set_query(QString const& query);
	bool select_type(QString const& qt_type);
	void enable_used_format(bool is_existing);
	void show_def_params();
	static void show_layout_children(const QLayout* layout, bool show);

private slots:
	void update_fields_list();
	void update_fields_list_format();

private:
	arcsight::ArcsightLogger connect_to_arcsight();
	void create_new_format();
	void set_list_fields_kept();

protected:
	QSettings _settings;
	int64_t _last_load_preset;
	PVInputTypeArcsight const* _in_t;

protected:
	QPushButton* _btn_sqlite_browse;

protected:
	// New XML format created from the database fields
	QDomDocument _new_format_doc;
	std::vector<QString> _fields_name;
	PVArcsightQuery::fields_indexes_t _fields_kept;
};

}

#endif
