/**
 * \file PVElasticsearchParamsWidget.cpp
 *
 * Copyright (C) Picviz Labs 2015
 */

#include "PVElasticsearchParamsWidget.h"
#include "PVInputTypeElasticsearch.h"

#include <pvkernel/core/PVProgressBox.h>

#include <QMessageBox>
#include <QPushButton>

enum EQueryType {
	QUERY_BUILDER = 0,
	JSON,
	SQL,

	QUERY_TYPE_COUNT
};
static const char* query_types[] = { "Query Builder", "JSON", "SQL" };

PVRush::PVElasticsearchParamsWidget::PVElasticsearchParamsWidget(
	PVInputTypeElasticsearch const* in_t,
	PVRush::hash_formats const& formats,
	QWidget* parent
) : PVParamsWidget<PVInputTypeElasticsearch, PVElasticsearchPresets, PVElasticsearchInfos, PVElasticsearchQuery>(in_t, formats, parent)
{
	QLabel* label_index = new QLabel("Index :");
	_btn_refresh = new QPushButton("&Refresh");
	_combo_index = new QComboBox();
	_port_sb->setValue(PVElasticsearchAPI::DEFAULT_PORT);

	_btn_refresh->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Minimum);
	_combo_index->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Minimum);

	QHBoxLayout* custom_layout = new QHBoxLayout();
	custom_layout->addWidget(label_index);
	custom_layout->addWidget(_combo_index);
	custom_layout->addWidget(_btn_refresh);

	_custom_layout->addLayout(custom_layout);

	connect(_combo_index, SIGNAL(currentIndexChanged(const QString&)), this, SLOT(index_changed_slot(const QString&)));
	connect(_combo_index, SIGNAL(activated(int)), this, SLOT(index_changed_by_user_slot()));
	connect(_btn_refresh, SIGNAL(clicked()), this, SLOT(fetch_server_data_slot()));

	for (size_t i = 0 ; i < QUERY_TYPE_COUNT ; i++) {
		_query_type_cb->addItem(query_types[i]);
	}

	_help_label->setText(
	"<html>"
		"<head/>"
			"<body>"
				"<p><span style=\" text-decoration: underline;\">Elasticsearch limitation:</span><br/></p>"
				"<p>The order of the lines returned by Elasticsearch queries does not respect the order of the lines as imported by Logstash. "
				"</p><br/>"
				"<p><span style=\" text-decoration: underline;\">Logstash CSV plugin limitation:</span><br/></p>"
				"<p>When importing logs into Elasticsearch using Logstash CSV plugin, be careful not to name any of your columns "
				"<span style=\" font-weight:600;\">message</span>, <span style=\" font-weight:600;\">type</span>,"
				"<span style=\" font-weight:600;\">host</span>,<span style=\" font-weight:600;\"> path</span> or"
				"<span style=\" font-weight:600;\">geoip </span>because these are internally used by Logstash and will likely to cause conflicts.</p>"
			"</body>"
	"</html>");
}

QString PVRush::PVElasticsearchParamsWidget::get_sql_query_prefix() const
{
	return QString("SELECT * FROM %1 WHERE ").arg(get_infos().get_index());
}

void PVRush::PVElasticsearchParamsWidget::index_changed_slot(const QString& index)
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

void PVRush::PVElasticsearchParamsWidget::fetch_server_data_slot()
{
	fetch_server_data(get_infos());
}

bool PVRush::PVElasticsearchParamsWidget::check_connection(std::string* error /*= nullptr*/)
{
	const PVElasticsearchInfos& infos = get_infos();

	PVRush::PVElasticsearchAPI es(get_infos());

	bool ret = es.check_connection(error);

	if (ret) {
		fetch_server_data(infos);
	}

	return ret;
}

void PVRush::PVElasticsearchParamsWidget::query_type_changed_slot()
{
	int query_type = _query_type_cb->currentIndex();

	_querybuilder->setVisible(false);
	_txt_query->setVisible(false);

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
		_txt_query->setVisible(true);
	}
	else if (query_type == EQueryType::QUERY_BUILDER) {
		_gb_query->setTitle("Query");
		_reference_label->setText("");
		_querybuilder->reset_rules();
		_querybuilder->setVisible(true);
	}
	else { // EQueryType::JSON
		_txt_query->setPlainText("{ \"query\" : { \"match_all\" : { } } }");
		_gb_query->setTitle("Query");
		_reference_label->setText(
			"<a href=\"https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-filters.html\">"
			"<span style=\" text-decoration: underline; color:#0000ff;\">Elasticsearch Filters reference"
		);
		_txt_query->setVisible(true);
		_txt_query->setEnabled(true);
	}
}

void PVRush::PVElasticsearchParamsWidget::export_query_result(QTextStream& output_stream, PVCore::PVProgressBox& pbox, std::string* error /*= nullptr*/)
{
	size_t count = 0;
	bool query_end = false;

	PVRush::PVElasticsearchAPI es(get_infos());
	const PVElasticsearchQuery& query = get_query(error);
	PVElasticsearchAPI::rows_chunk_t rows_array;

	size_t max_count = es.scroll_count();
	pbox.getProgressBar()->setMaximum(max_count);

	do {
		query_end = es.extract(query, rows_array, error);

		if (error && error->empty() == false) {
			return;
		}

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

		if (output_stream.status() == QTextStream::WriteFailed) {
			if (error) {
				*error = "Write failed. Is your disk full ?";
			}
		}

		pbox.set_status(count);
	} while (query_end == false);
}

bool PVRush::PVElasticsearchParamsWidget::set_infos(PVElasticsearchInfos const& infos)
{
	bool res = PVParamsWidget<PVInputTypeElasticsearch, PVElasticsearchPresets, PVElasticsearchInfos, PVElasticsearchQuery>::set_infos(infos);

	_combo_index->setCurrentIndex(_combo_index->findText(infos.get_index()));

	return res;
}

PVRush::PVElasticsearchInfos PVRush::PVElasticsearchParamsWidget::get_infos() const
{
	PVRush::PVElasticsearchInfos infos = PVParamsWidget<PVInputTypeElasticsearch, PVElasticsearchPresets, PVElasticsearchInfos, PVElasticsearchQuery>::get_infos();
	infos.set_index(_combo_index->currentText());

	return infos;
}

QString PVRush::PVElasticsearchParamsWidget::get_server_query(std::string* error /* = nullptr */) const
{
	QString q = get_serialize_query();
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::JSON) {
		q = _txt_query->toPlainText();
	}
	else {
		PVRush::PVElasticsearchAPI es(get_infos());

		if (query_type == EQueryType::SQL) {
			q = es.sql_to_json(QString(get_sql_query_prefix() + q).toStdString(), error).c_str();
		} else if (query_type == EQueryType::QUERY_BUILDER) {
			q = QString::fromStdString(es.rules_to_json(_querybuilder->get_rules()));
		}
	}

	while(q.endsWith('\n')) q.chop(1);

	return q;
}

QString PVRush::PVElasticsearchParamsWidget::get_serialize_query() const
{
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::QUERY_BUILDER) {
		return QString(_querybuilder->get_rules().c_str());
	}
	else {
		return _txt_query->toPlainText();
	}
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

size_t PVRush::PVElasticsearchParamsWidget::query_result_count(std::string* error /* = nullptr */)
{
	std::string err;
	PVRush::PVElasticsearchAPI es(get_infos());
	const PVElasticsearchQuery& query = get_query(&err);

	if (err.empty() == false) {
		if (error) {
			*error = err;
		}
		return 0;
	}

	return es.count(query, error);
}

bool PVRush::PVElasticsearchParamsWidget::fetch_server_data(const PVElasticsearchInfos& infos)
{
	PVRush::PVElasticsearchAPI es(infos);
	PVRush::PVElasticsearchAPI::indexes_t indexes = es.indexes();

	QString old_index = _combo_index->currentText();

	_combo_index->clear();
	for (const std::string& index : indexes) {
		_combo_index->addItem(QString(index.c_str()));
	}

	_combo_index->setCurrentIndex(_combo_index->findText(old_index));

	return true;
}
