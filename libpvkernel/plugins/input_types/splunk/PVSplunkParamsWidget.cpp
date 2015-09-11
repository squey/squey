#include "PVSplunkParamsWidget.h"
#include "PVInputTypeSplunk.h"
#include "../../common/splunk/PVSplunkAPI.h"
#include "../../common/splunk/PVSplunkJsonConverter.h"

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/widgets/PVQueryBuilder.h>

#include <QMessageBox>
#include <QPushButton>
#include <QFileDialog>

static const char* query_types[] = { "Query Builder", "Splunk search API" };
static const size_t UNCORRECT_COUNT = -1;

PVRush::PVSplunkParamsWidget::PVSplunkParamsWidget(PVInputTypeSplunk const* in_t, PVRush::hash_formats const& formats, QWidget* parent)
	: PVParamsWidget<PVInputTypeSplunk, PVSplunkPresets, PVSplunkInfos, PVSplunkQuery>(in_t, formats, parent)
{
	QHBoxLayout* custom_layout = new QHBoxLayout();
	_custom_layout->addLayout(custom_layout);

	auto setup_combo = [&](QComboBox** cb, const QString& l)
	{
		QLabel* label = new QLabel(l);
		(*cb) = new QComboBox();
		QObject::connect(*cb, (void (QComboBox::*)(int)) &QComboBox::activated, this, &PVSplunkParamsWidget::splunk_filter_changed_by_user_slot);
		(*cb)->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Minimum);
		custom_layout->addWidget(label);
		custom_layout->addWidget(*cb);
	};

	setup_combo(&_combo_index, "Index :");
	setup_combo(&_combo_host, "Host :");
	setup_combo(&_combo_sourcetype, "Sourcetype :");

	_gb_query->setTitle("Query");

	_port_sb->setValue(PVSplunkAPI::DEFAULT_PORT);

	for (const char * const qtype_name: query_types) {
		_query_type_cb->addItem(qtype_name);
	}

	_help_label->setText(
	"<html>"
		"<head/>"
			"<body>"
				"<p><span style=\" text-decoration: underline;\">Splunk limitation:</span><br/></p>"
				"<p>The order of the lines returned by Splunk queries does not respect the order of the lines as imported. "
				"</p>"
			"</body>"
	"</html>");
}

bool PVRush::PVSplunkParamsWidget::check_connection(std::string* error /*= nullptr*/)
{
	const PVSplunkInfos& infos = get_infos();

	PVRush::PVSplunkAPI splunk(infos);

	bool ret = splunk.check_connection(error);

	if (ret) {
		fetch_server_data(infos);
	}

	return ret;
}

void PVRush::PVSplunkParamsWidget::query_type_changed_slot()
{
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::QUERY_BUILDER) {
		_reference_label->setText("");
		_querybuilder->reset_rules();
		_querybuilder->setVisible(true);
		_txt_query->setVisible(false);
	}
	else { // EQueryType::SPLUNK
		_txt_query->setPlainText("");
		_reference_label->setText(
			"<a href=\"http://docs.splunk.com/Documentation/Splunk/latest/RESTUM/RESTusing\">"
			"<span style=\" text-decoration: underline; color:#0000ff;\">Splunk REST API"
		);
		_querybuilder->setVisible(false);
		_txt_query->setVisible(true);
		_txt_query->setEnabled(true);
	}
}

void PVRush::PVSplunkParamsWidget::export_query_result(QTextStream& output_stream, PVCore::PVProgressBox& pbox, std::string* error /*= nullptr*/)
{
	bool query_end = false;

	PVRush::PVSplunkAPI splunk(get_infos());
	const PVSplunkQuery& query = get_query(error);
	std::string data;

	do {
		query_end = !splunk.extract(query, data, error);

		if (error && error->empty() == false) {
			return;
		}

		if (pbox.get_cancel_state() == PVCore::PVProgressBox::CANCEL ||
			pbox.get_cancel_state() == PVCore::PVProgressBox::CANCEL2) {
			break;
		}

		output_stream << data.c_str() << endl;

		if (output_stream.status() == QTextStream::WriteFailed) {
			if (error) {
				*error = "Write failed. Is your disk full ?";
			}
		}

	}  while (query_end == false);
}

bool PVRush::PVSplunkParamsWidget::set_infos(PVSplunkInfos const& infos)
{

	bool res = PVParamsWidget<PVInputTypeSplunk, PVSplunkPresets, PVSplunkInfos, PVSplunkQuery>::set_infos(infos);

	_combo_index->setCurrentIndex(_combo_index->findText(infos.get_splunk_index()));
	_combo_host->setCurrentIndex(_combo_host->findText(infos.get_splunk_host()));
	_combo_sourcetype->setCurrentIndex(_combo_sourcetype->findText(infos.get_splunk_sourcetype()));

	return res;
}

PVRush::PVSplunkInfos PVRush::PVSplunkParamsWidget::get_infos() const
{
	PVRush::PVSplunkInfos infos = PVParamsWidget<PVInputTypeSplunk, PVSplunkPresets, PVSplunkInfos, PVSplunkQuery>::get_infos();

	infos.set_splunk_index(_combo_index->currentText());
	infos.set_splunk_host(_combo_host->currentText());
	infos.set_splunk_sourcetype(_combo_sourcetype->currentText());

	return infos;
}

QString PVRush::PVSplunkParamsWidget::get_server_query(std::string* /*error  = nullptr */) const
{
	QString q = get_serialize_query();
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::QUERY_BUILDER) {
		PVRush::PVSplunkAPI splunk(get_infos());
		q = QString::fromStdString(splunk.rules_to_json(_querybuilder->get_rules()));
	}
	else {
		q = _txt_query->toPlainText();
	}

	while(q.endsWith('\n')) q.chop(1);

	return q;
}

QString PVRush::PVSplunkParamsWidget::get_serialize_query() const
{
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::QUERY_BUILDER) {
		return QString(_querybuilder->get_rules().c_str());
	}
	else {
		return _txt_query->toPlainText();
	}
}

void PVRush::PVSplunkParamsWidget::set_query(QString const& query)
{
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::QUERY_BUILDER) {
		PVRush::PVSplunkAPI splunk(get_infos());
		_querybuilder->set_filters(splunk.columns());
		_querybuilder->set_rules(query.toStdString());
	}
	else {
		_txt_query->setPlainText(query);
	}
}

size_t PVRush::PVSplunkParamsWidget::query_result_count(std::string* error /* = nullptr */)
{
	std::string err;
	PVRush::PVSplunkAPI splunk(get_infos());
	const PVSplunkQuery& query = get_query(&err);

	if (err.empty() == false) {
		if (error) {
			*error = err;
		}
		return UNCORRECT_COUNT;
	}

	return splunk.count(query, error);
}

void PVRush::PVSplunkParamsWidget::splunk_filter_changed_by_user_slot()
{
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::QUERY_BUILDER) {
		PVRush::PVSplunkAPI splunk(get_infos());
		_querybuilder->set_filters(splunk.columns());
	}
}

bool PVRush::PVSplunkParamsWidget::fetch_server_data(const PVSplunkInfos& infos)
{
	PVRush::PVSplunkAPI splunk(infos);

	auto refresh = [](QComboBox* cb, const PVRush::PVSplunkAPI::strings_t& list)
	{
		QString old_item = cb->currentText();
		cb->clear();
		for (const std::string& item : list) {
			cb->addItem(QString(item.c_str()));
		}
		cb->setCurrentIndex(cb->findText(old_item));
	};

	refresh(_combo_index, splunk.indexes());
	refresh(_combo_host, splunk.hosts());
	refresh(_combo_sourcetype, splunk.sourcetypes());

	return true;
}
