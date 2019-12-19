/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include "PVOpcUaParamsWidget.h"
#include "PVInputTypeOpcUa.h"
#include "../../common/opcua/PVOpcUaTreeModel.h"
#include "../../common/opcua/PVOpcUaAPI.h"

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/widgets/PVFilterableComboBox.h>
#include <pvkernel/widgets/PVQueryBuilder.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVFormat_types.h>

#include <QMessageBox>
#include <QPushButton>
#include <QDesktopWidget>
#include <QOpcUaClient>
#include <QOpcUaProvider>
#include <QOpcUaPkiConfiguration>
#include <QOpcUaAuthenticationInformation>
#include <QOpcUaErrorState>

// #include <QSslSocket>
// #include <openssl/ssl.h>

static const char* query_types[] = {"Query Builder", "JSON", "SQL"};
static constexpr const char MATCH_ALL_QUERY[] = R"###({ "query" : { "match_all" : { } } })###";

PVRush::PVOpcUaParamsWidget::PVOpcUaParamsWidget(PVInputTypeOpcUa const* in_t,
                                                 PVRush::hash_formats const& formats,
                                                 QWidget* parent)
    : PVParamsWidget<PVInputTypeOpcUa, PVOpcUaPresets, PVOpcUaInfos, PVOpcUaQuery>(
          in_t, formats, parent)
{
	QLabel* label_index = new QLabel("Index/alias :");
	_btn_refresh = new QPushButton("&Refresh");
	_combo_index = new PVWidgets::PVFilterableComboBox();
	//_port_sb->setValue(PVOpcUaAPI::DEFAULT_PORT);

	_btn_refresh->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Minimum);
	_combo_index->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Minimum);

	// QHBoxLayout* custom_layout = new QHBoxLayout();
	// custom_layout->addWidget(label_index);
	// custom_layout->addWidget(_combo_index);
	// custom_layout->addWidget(_btn_refresh);

	// _custom_layout->addLayout(custom_layout);

	_columns_tree_widget->header()->setSectionResizeMode(QHeaderView::ResizeToContents);

	connect(_combo_index,
	        static_cast<void (QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged), this,
	        &PVOpcUaParamsWidget::index_changed_slot);
	connect(_combo_index, static_cast<void (QComboBox::*)(int)>(&QComboBox::activated), this,
	        &PVOpcUaParamsWidget::index_changed_by_user_slot);
	connect(_btn_refresh, &QPushButton::clicked, this,
	        &PVOpcUaParamsWidget::fetch_server_data_slot);

	for (const char* const qtype_name : query_types) {
		_query_type_cb->addItem(qtype_name);
	}

	_help_label->setText(
	    "<html>"
	    "<head/>"
	    "<body>"
	    "<p><span style=\" text-decoration: underline;\">OpcUa limitation:</span><br/></p>"
	    "<p><ul><li>The order of the lines returned by OpcUa queries does not respect the "
	    "order of the lines as imported by Logstash.</li>"
	    "<li>Requests use tokenized strings. If filters were applied during import, the search "
	    "have to respect their constraints.<br/>"
	    "<i>Eg : If a lowercase filter was used at import time, the according search must be done "
	    "in lowercase too.</i></li></ul>"
	    "</p><br/>"
	    "<p><span style=\" text-decoration: underline;\">Logstash CSV plugin "
	    "limitation:</span><br/></p>"
	    "<p>When importing logs into OpcUa using Logstash CSV plugin, be careful not to "
	    "name any of your columns "
	    "<span style=\" font-weight:600;\">message</span>, <span style=\" "
	    "font-weight:600;\">type</span>, "
	    "<span style=\" font-weight:600;\">host</span>,<span style=\" font-weight:600;\"> "
	    "path</span> or "
	    "<span style=\" font-weight:600;\">geoip </span>because these are internally used by "
	    "Logstash and will likely to cause conflicts.</p>"
	    "</body>"
	    "</html>");

	setFixedHeight(QApplication::desktop()->availableGeometry().height() - 50);
}

QString PVRush::PVOpcUaParamsWidget::get_sql_query_prefix() const
{
	return QString("SELECT * FROM %1 WHERE ").arg(get_infos().get_index());
}

void PVRush::PVOpcUaParamsWidget::index_changed_slot(const QString& index)
{
	buttonBox->buttons()[0]->setEnabled(index != "");

	QString query_type = _query_type_cb->currentText();

	if (query_type == query_types[EQueryType::SQL]) {
		_gb_query->setTitle(get_sql_query_prefix() + " ...");
	}
}

void PVRush::PVOpcUaParamsWidget::reset_columns_tree_widget()
{
	disconnect(_columns_tree_widget, &QTreeWidget::itemChanged, this,
	           &PVOpcUaParamsWidget::tree_item_changed);

	qDebug() << "reset_columns_tree_widget";

	_columns_tree_widget->clear();
	_root_item = new QTreeWidgetItem(_columns_tree_widget);
	_root_item->setExpanded(true);
	_root_item->setText(0, "properties");
	_root_item->setCheckState(0, Qt::Unchecked);

	for (auto& opcua_column : _opcua_treeview->selectionModel()->selectedRows()) {
		auto column_node_id = opcua_column.siblingAtColumn(4).data();
		auto column_name = opcua_column.siblingAtColumn(5).data().value<QString>();
		auto column_type = opcua_column.siblingAtColumn(3).data().value<QString>();
		qDebug() << "Selected NodeId:" << opcua_column.siblingAtColumn(4).data()
		         << "ColumnName:" << column_name << "ColumnType:" << column_type;
		QTreeWidgetItem* tree_item = new QTreeWidgetItem(_root_item);
		tree_item->setText(0, "(" + column_node_id.value<QString>() + ") " + column_name);
		tree_item->setData(0, Qt::UserRole, column_node_id);
		tree_item->setText(1, column_type);
		tree_item->setExpanded(true);
		tree_item->setCheckState(0, Qt::Checked);
	}

	// std::vector<QTreeWidgetItem*> parents({_root_item});

	// PVRush::PVOpcUaAPI es(get_infos());
	// es.visit_columns([&](const std::string& rel_name, const std::string& abs_name,
	//                      const std::string& type, bool is_leaf, bool is_last_child) {
	// 	QTreeWidgetItem* tree_item = new QTreeWidgetItem(parents.back());

	// 	tree_item->setText(0, rel_name.c_str());
	// 	tree_item->setData(0, Qt::UserRole, QString::fromStdString(abs_name));
	// 	tree_item->setText(1, type.c_str());
	// 	tree_item->setExpanded(true);
	// 	tree_item->setCheckState(0, Qt::Unchecked);

	// 	if (not is_leaf) { // node
	// 		parents.emplace_back(tree_item);
	// 	}

	// 	if (is_last_child) {
	// 		parents.pop_back();
	// 	}
	// });

	_columns_tree_widget->resizeColumnToContents(0);

	connect(_columns_tree_widget, &QTreeWidget::itemChanged, this,
	        &PVOpcUaParamsWidget::tree_item_changed);
}

/*
 * Recursively visit all the columns of the columns tree widget
 *
 * @param tree_item root item to visit
 * @param f the function that is called against each QTreeWidgetItem
 */
static void visit_columns(QTreeWidgetItem* tree_item,
                          const std::function<void(QTreeWidgetItem*)>& f)
{
	f(tree_item);
	for (int i = 0; i < tree_item->childCount(); i++) {
		QTreeWidgetItem* tree_child = tree_item->child(i);
		visit_columns(tree_child, f);
	}
}

/*
 * Recursively visit the checked columns of the columns tree widget
 *
 * @param tree_item root item to visit
 * @param f the function that is called against each QTreeWidgetItem
 */
static void visit_selected_columns(QTreeWidgetItem* tree_item,
                                   const std::function<void(QTreeWidgetItem*)>& f)
{
	if (not tree_item)
		return;

	static auto is_checked = [](const QTreeWidgetItem* item) {
		return item->checkState(0) == Qt::Checked or item->checkState(0) == Qt::PartiallyChecked;
	};

	if (is_checked(tree_item) and tree_item->childCount() == 0) {
		f(tree_item);
	}

	for (int i = 0; i < tree_item->childCount(); i++) {
		QTreeWidgetItem* tree_child = tree_item->child(i);
		if (is_checked(tree_child)) {
			visit_selected_columns(tree_child, f);
		}
	}
}

static void propagate_check_state(QTreeWidgetItem* leaf_item)
{
	QTreeWidgetItem* parent = leaf_item->parent();
	if (parent == nullptr) {
		return;
	}

	int child_count = parent->childCount();
	int checked_count = 0;
	for (int i = 0; i < child_count; i++) {
		checked_count += parent->child(i)->checkState(0);
	}
	Qt::CheckState parent_state =
	    (checked_count == child_count * 2
	         ? Qt::Checked
	         : (checked_count == 0 ? Qt::Unchecked : Qt::PartiallyChecked));

	parent->setCheckState(0, parent_state);
	propagate_check_state(parent);
}

void PVRush::PVOpcUaParamsWidget::tree_item_changed(QTreeWidgetItem* item, int column)
{
	_columns_tree_widget->blockSignals(true);

	if (column == 0) {
		if (item->childCount() > 0) { // node
			Qt::CheckState check_state = item->checkState(0);
			visit_columns(item,
			              [&](QTreeWidgetItem* child) { child->setCheckState(0, check_state); });
		}
		propagate_check_state(item);
	}

	_columns_tree_widget->blockSignals(false);
}

void PVRush::PVOpcUaParamsWidget::index_changed_by_user_slot()
{
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::QUERY_BUILDER) {
		// PVRush::PVOpcUaAPI es(get_infos());
		// _querybuilder->set_filters(es.querybuilder_columns());
	}

	reset_columns_tree_widget();
}

void PVRush::PVOpcUaParamsWidget::fetch_server_data_slot()
{
	fetch_server_data(get_infos());
}

bool PVRush::PVOpcUaParamsWidget::check_connection(std::string* error /*= nullptr*/)
{
	const PVOpcUaInfos& infos = get_infos();

	// PVRush::PVOpcUaAPI es(infos);

	// bool ret = es.check_connection(error);

	// if (ret) {
	// 	fetch_server_data(infos);
	// }

	// return ret;

	// SSL_library_init();
	// SSL_load_error_strings();
	// ERR_load_BIO_strings();
	// OpenSSL_add_all_algorithms();
	// OpenSSL_add_all_digests();

	// qDebug() << "sslLibraryVersionString=" << QSslSocket::sslLibraryVersionString();
	// qDebug() << "supportsSsl=" << QSslSocket::supportsSsl();

	// return false;

	QOpcUaProvider provider;
	if (provider.availableBackends().isEmpty()) {
		qDebug() << "No OpcUa backend available!";
		return false;
	}
	QOpcUaClient* client = provider.createClient(provider.availableBackends()[0]);
	if (!client) {
		qDebug() << "OpcUa backend (" << provider.availableBackends()[0]
		         << ") could not be loaded and could not create client.";
		return false;
	}

	QString pkidir("/home/fchapelle/dev/qtopcua/lay2form/pkidir/");

	QOpcUaPkiConfiguration pkiConfig;
	pkiConfig.setClientCertificateFile(pkidir + "/own/certs/lay2form_fchapelle_certificate.der");
	pkiConfig.setPrivateKeyFile(pkidir + "/own/private/lay2form_fchapelle_privatekey.pem");
	pkiConfig.setTrustListDirectory(pkidir + "/trusted/certs");
	pkiConfig.setRevocationListDirectory(pkidir + "/trusted/crl");
	pkiConfig.setIssuerListDirectory(pkidir + "/issuers/certs");
	pkiConfig.setIssuerRevocationListDirectory(pkidir + "/issuers/crl");

	QOpcUaAuthenticationInformation authInfo;
	if (infos.get_login().isEmpty()) {
		authInfo.setAnonymousAuthentication();
	} else {
		authInfo.setUsernameAuthentication(infos.get_login(), infos.get_password());
	}

	client->setAuthenticationInformation(authInfo);
	client->setPkiConfiguration(pkiConfig);
	client->setApplicationIdentity(pkiConfig.applicationIdentity());

	QObject::connect(
	    client, &QOpcUaClient::stateChanged, [client, this](QOpcUaClient::ClientState state) {
		    qDebug() << "Client state changed:" << state;
		    if (state == QOpcUaClient::ClientState::Connected) {
			    QOpcUaNode* node = client->node("ns=0;i=84");
			    if (node) {
				    qDebug() << "A node object has been created";
			    }
			    _opcua_treeview = new QTreeView(this);
			    _opcua_treeview->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
			    _opcua_treeview->setTextElideMode(Qt::ElideRight);
			    _opcua_treeview->setAlternatingRowColors(true);
			    _opcua_treeview->setSelectionBehavior(QAbstractItemView::SelectRows);
			    _opcua_treeview->setSelectionMode(QAbstractItemView::MultiSelection);
			    _opcua_treeview->setMinimumWidth(width());
			    _opcua_treeview->setMinimumHeight(height());
			    auto treemodel = new PVOpcUaTreeModel(this);
			    treemodel->setOpcUaClient(client);
			    _opcua_treeview->setModel(treemodel);
			    _opcua_treeview->header()->setSectionResizeMode(1 /* Value column*/,
			                                                    QHeaderView::Interactive);
			    tabWidget->addTab(_opcua_treeview, "Browse nodes");
			    connect(_opcua_treeview->selectionModel(), &QItemSelectionModel::selectionChanged,
			            [this](const QItemSelection& selected, const QItemSelection& deselected) {
				            reset_columns_tree_widget();
				            for (auto& selitem : selected) {
					            //
				            }
			            });
		    }
	    });

	connect(client, &QOpcUaClient::connectError, [](QOpcUaErrorState* errorState) {
		qDebug() << "Client Error State:" << QOpcUa::statusToString(errorState->errorCode());
		if (errorState->isClientSideError() &&
		    errorState->connectionStep() ==
		        QOpcUaErrorState::ConnectionStep::CertificateValidation) {
			errorState->setIgnoreError(true);
		}
	});

	QObject::connect(client, &QOpcUaClient::endpointsRequestFinished,
	                 [client](QVector<QOpcUaEndpointDescription> endpoints,
	                          QOpcUa::UaStatusCode statusCode, QUrl requestUrl) {
		                 qDebug() << "Endpoints returned:" << endpoints.count() << statusCode
		                          << requestUrl;
		                 if (endpoints.size()) {
			                 endpoints.first().setEndpointUrl(
			                     requestUrl.toString()); // Needed to workaround an unconfigured
			                                             // reverse DNS.
			                 client->connectToEndpoint(
			                     endpoints.first()); // Connect to the first endpoint in the list
		                 }
	                 });

	if (client->requestEndpoints(QUrl(infos.get_host()))) {
		qDebug() << "OpcUa client requesting endpoints...";
		return true;
	} else {
		qDebug() << "OpcUa client could not request endpoints.";
	}

	return false;
}

void PVRush::PVOpcUaParamsWidget::query_type_changed_slot()
{
	// int query_type = _query_type_cb->currentIndex();

	// _querybuilder->setVisible(false);
	// _txt_query->setVisible(false);

	// if (query_type == EQueryType::SQL) {
	// 	_txt_query->setPlainText("");
	// 	_reference_label->setText(
	// 	    "<a href=\"https://github.com/NLPchina/elasticsearch-sql/\">"
	// 	    "<span style=\" text-decoration: underline; color:#0000ff;\">OpcUa SQL plugin");
	// 	PVRush::PVOpcUaAPI es(get_infos());
	// 	if (es.is_sql_available()) {
	// 		_gb_query->setTitle(get_sql_query_prefix() + " ...");
	// 		_txt_query->setEnabled(true);
	// 	} else {
	// 		_txt_query->setPlainText("Please, install the SQL plugin to your OpcUa "
	// 		                         "instance to support this feature.");
	// 		_txt_query->setEnabled(false);
	// 		// buttonBox->buttons()[0]->setEnabled(false);
	// 	}
	// 	_txt_query->setVisible(true);
	// } else if (query_type == EQueryType::QUERY_BUILDER) {
	// 	_gb_query->setTitle("Query");
	// 	_reference_label->setText("");
	// 	_querybuilder->reset_rules();

	// 	// PVRush::PVOpcUaAPI es(get_infos());
	// 	// _querybuilder->set_filters(es.querybuilder_columns());

	// 	_querybuilder->setVisible(true);
	// } else { // EQueryType::JSON
	// 	_txt_query->setPlainText(MATCH_ALL_QUERY);
	// 	_gb_query->setTitle("Query");
	// 	_reference_label->setText("<a "
	// 	                          "href=\"https://www.elastic.co/guide/en/elasticsearch/reference/"
	// 	                          "current/query-dsl-filters.html\">"
	// 	                          "<span style=\" text-decoration: underline; "
	// 	                          "color:#0000ff;\">OpcUa Filters reference");
	// 	_txt_query->setVisible(true);
	// 	_txt_query->setEnabled(true);
	// }
}

/*****************************************************************************
 * PVRush::PVOpcUaParamsWidget::get_export_filters
 *****************************************************************************/

QString PVRush::PVOpcUaParamsWidget::get_export_filters()
{
	return "CSV File (*.csv)";
}

void PVRush::PVOpcUaParamsWidget::export_query_result(PVCore::PVStreamingCompressor& compressor,
                                                      const std::string& sep,
                                                      const std::string& quote,
                                                      bool header,
                                                      PVCore::PVProgressBox& pbox,
                                                      std::string* error)
{
	size_t count = 0;
	bool query_end = false;

	PVRush::PVOpcUaAPI es(get_infos());
	const PVOpcUaQuery& query = get_query(error);

	using PVRush::PVUtils::safe_export;

	try {
		if (header) {
			compressor.write(safe_export("Source Timestamp", sep, quote) + sep +
			                 safe_export("Data", sep, quote) + "\n");
		}

		// pbox.set_maximum(es.count(query));

		for (auto& opcua_column : _opcua_treeview->selectionModel()->selectedRows()) {

			if (error && error->empty() == false) {
				return;
			}

			if (pbox.get_cancel_state() == PVCore::PVProgressBox::CancelState::CANCEL ||
			    pbox.get_cancel_state() == PVCore::PVProgressBox::CancelState::CANCEL2) {
				break;
			}

			auto column_node_id =
			    opcua_column.siblingAtColumn(0).data(Qt::UserRole).value<QString>();
			es.read_node_history(
			    column_node_id, UA_DateTime_fromUnixTime(0), UA_DateTime_now(),
			    [&compressor, &sep, &quote, &count](UA_HistoryData* data) {
				    for (UA_UInt32 i = 0; i < data->dataValuesSize; ++i) {
					    UA_DataValue& value = data->dataValues[i];
					    std::string v;
					    v += safe_export(std::to_string(value.sourceTimestamp), sep, quote) + sep;
					    v += safe_export(std::to_string(value.serverTimestamp), sep, quote) + sep;
					    v +=
					        safe_export(PVOpcUaAPI::to_json_string(value.value), sep, quote) + "\n";
					    compressor.write(v);
				    }
				    count += data->dataValuesSize;
				    return true;
			    });

			pbox.set_value(count);
			pbox.set_extended_status(QString("%L1 rows exported so far").arg(count));
		}
	} catch (const PVCore::PVStreamingCompressorError& e) {
		*error = e.what();
	}
}

bool PVRush::PVOpcUaParamsWidget::set_infos(PVOpcUaInfos const& infos)
{
	bool res =
	    PVParamsWidget<PVInputTypeOpcUa, PVOpcUaPresets, PVOpcUaInfos, PVOpcUaQuery>::set_infos(
	        infos);
	_combo_index->setCurrentIndex(_combo_index->findText(infos.get_index()));

	if (infos.get_index().isEmpty()) {
		return true;
	}

	// update filter_path
	reset_columns_tree_widget();
	const std::string& filter_path = infos.get_filter_path().toStdString();
	std::unordered_set<std::string> selected_columns;
	boost::split(selected_columns, filter_path, boost::is_any_of(","));

	visit_columns(_root_item, [&](QTreeWidgetItem* item) {
		const std::string& abs_name = item->data(0, Qt::UserRole).toString().toStdString();
		if (selected_columns.find(abs_name) != selected_columns.end()) {
			item->setCheckState(0, Qt::Checked);
		}
	});

	return res;
}

PVRush::PVOpcUaInfos PVRush::PVOpcUaParamsWidget::get_infos() const
{
	PVRush::PVOpcUaInfos infos =
	    PVParamsWidget<PVInputTypeOpcUa, PVOpcUaPresets, PVOpcUaInfos, PVOpcUaQuery>::get_infos();

	infos.set_index(_combo_index->currentText());
	infos.set_filter_path(_format_path->text());

	// update filter_path
	if (_root_item) {
		std::vector<std::string> selected_columns;
		visit_selected_columns(_root_item, [&](const QTreeWidgetItem* item) {
			selected_columns.emplace_back(item->data(0, Qt::UserRole).toString().toStdString());
		});
		const std::string& filter_path = boost::algorithm::join(selected_columns, ",");
		infos.set_filter_path(QString::fromStdString(filter_path));
	}

	return infos;
}

QString PVRush::PVOpcUaParamsWidget::get_server_query(std::string* error /* = nullptr */) const
{
	QString q = get_serialize_query();
	int query_type = _query_type_cb->currentIndex();

	if (query_type == EQueryType::JSON) {
		q = _txt_query->toPlainText();
	} else {
		// PVRush::PVOpcUaAPI es(get_infos());

		// if (query_type == EQueryType::SQL) {
		// 	q = es.sql_to_json(QString(get_sql_query_prefix() + q).toStdString(), error).c_str();
		// } else if (query_type == EQueryType::QUERY_BUILDER) {
		// 	q = QString::fromStdString(es.rules_to_json(_querybuilder->get_rules()));
		// 	if (q.isEmpty()) {
		// 		q = MATCH_ALL_QUERY;
		// 	}
		// }
	}

	while (q.endsWith('\n'))
		q.chop(1);

	return q;
}

QString PVRush::PVOpcUaParamsWidget::get_serialize_query() const
{
	QString serialized;
	for (auto& opcua_column : _opcua_treeview->selectionModel()->selectedRows()) {
		auto column_node_id = opcua_column.siblingAtColumn(0).data(Qt::UserRole).value<QString>();
		auto column_type = opcua_column.siblingAtColumn(3).data(Qt::UserRole).value<QString>();
		auto column_name = opcua_column.siblingAtColumn(5).data(Qt::UserRole).value<QString>();
		serialized += column_node_id + ";$;" + column_type + ";$;" + column_name + ";$;";
	}
	return serialized;
}

void PVRush::PVOpcUaParamsWidget::set_query(QString const& query)
{
	qDebug() << __func__ << query;
}

size_t PVRush::PVOpcUaParamsWidget::query_result_count(std::string* error /* = nullptr */)
{
	return 0;
	// std::string err;
	// PVRush::PVOpcUaAPI es(get_infos());
	// const PVOpcUaQuery& query = get_query(&err);

	// if (err.empty() == false) {
	// 	if (error) {
	// 		*error = err;
	// 	}
	// 	return 0;
	// }

	// return es.count(query, error);
}

bool PVRush::PVOpcUaParamsWidget::fetch_server_data(const PVOpcUaInfos& infos)
{
	// PVRush::PVOpcUaAPI es(infos);

	// QString old_index = _combo_index->currentText();

	// _combo_index->clear();
	// QStringList indexes_and_alias_list;
	// for (const std::string& index : es.indexes()) {
	// 	indexes_and_alias_list << (index.c_str());
	// }
	// for (const std::string& alias : es.aliases()) {
	// 	indexes_and_alias_list << (alias.c_str());
	// }
	// _combo_index->set_string_list(indexes_and_alias_list);

	// _combo_index->setCurrentIndex(_combo_index->findText(old_index));

	return true;
}

size_t PVRush::PVOpcUaParamsWidget::get_selected_columns_count() const
{
	// size_t selected_columns_count = 0;

	// visit_selected_columns(_root_item,
	//                        [&](const QTreeWidgetItem* /*item*/) { selected_columns_count++; });

	// return selected_columns_count;

	return _opcua_treeview->selectionModel()->selectedRows().size();
}

static size_t get_first_level_fields_count(const QString& format_path)
{
	QDomDocument doc;

	QFile f(format_path);
	if (not f.open(QFile::ReadOnly | QFile::Text))
		return {};
	QTextStream in(&f);

	doc.setContent(in.readAll());

	const QDomElement& root = doc.documentElement();

	size_t fields_count = 0;
	for (QDomElement n = root.firstChildElement(); not n.isNull(); n = n.nextSiblingElement()) {
		fields_count += n.tagName() == PVFORMAT_XML_TAG_FIELD_STR;
	}

	return fields_count;
}

void PVRush::PVOpcUaParamsWidget::accept()
{
	size_t fields_count = get_selected_columns_count();
	if (fields_count < 2) {
		QMessageBox::critical(this, "Invalid format error",
		                      "At least two columns must be selected to have a valid format.");
		return;
	}

	if (is_format_custom()) {
		update_custom_format();
	} else if (not(fields_count == get_first_level_fields_count(_format_path->text()))) {
		QMessageBox::critical(
		    this, "Invalid format error",
		    "Number of columns mismatchs between existing format and selected mapping.\n\n"
		    "You may want to edit and save your custom format.");
		return;
	}

	QDialog::accept();
}

void PVRush::PVOpcUaParamsWidget::update_custom_format()
{
	// _custom_format = PVOpcUaAPI(get_infos()).get_format_from_mapping();
	std::unique_ptr<PVXmlTreeNodeDom> format_root(
	    PVRush::PVXmlTreeNodeDom::new_format(_custom_format));

	PVRush::PVXmlTreeNodeDom* time_node =
	    format_root->addOneField(QString("SourceTime"), QString("time"));
	time_node->setAttribute(QString(PVFORMAT_AXIS_TYPE_FORMAT_STR), "yyyy-MM-d H:m:ss.S");

	for (auto& opcua_column : _opcua_treeview->selectionModel()->selectedRows()) {
		auto column_name = opcua_column.siblingAtColumn(5).data().value<QString>();
		auto column_type = opcua_column.siblingAtColumn(3).data(Qt::UserRole).value<QString>();
		qDebug() << "Selected NodeId:" << opcua_column.siblingAtColumn(4).data()
		         << "ColumnName:" << column_name << "ColumnType:" << column_type;
		auto node_id_open62541 = PVRush::PVOpcUaAPI::NodeId(column_type).open62541();
		if (auto* data_type = UA_findDataType(&node_id_open62541)) {
			PVRush::PVXmlTreeNodeDom* node =
			    format_root->addOneField(column_name, QString(PVRush::PVOpcUaAPI::pvcop_type(data_type->typeIndex)));
		} else {
			PVRush::PVXmlTreeNodeDom* node = format_root->addOneField(column_name, QString("string"));
		}
	}
}

void PVRush::PVOpcUaParamsWidget::edit_custom_format()
{
	update_custom_format();
	PVParamsWidgetBase::edit_custom_format();
}
