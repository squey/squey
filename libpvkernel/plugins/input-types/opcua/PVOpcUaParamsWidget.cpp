//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVOpcUaParamsWidget.h"
#include "PVInputTypeOpcUa.h"
#include "../../common/opcua/PVOpcUaTreeModel.h"
#include "../../common/opcua/PVOpcUaAPI.h"

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/widgets/PVFilterableComboBox.h>
#include <pvkernel/widgets/PVQueryBuilder.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVFormat_types.h>

#include <QMenu>
#include <QMessageBox>
#include <QPushButton>
#include <QScrollArea>
#include <QOpcUaClient>
#include <QOpcUaProvider>
#include <QOpcUaPkiConfiguration>
#include <QOpcUaAuthenticationInformation>
#include <QOpcUaErrorState>
#include <QScreen>

// #include <QSslSocket>
// #include <openssl/ssl.h>

PVRush::PVOpcUaParamsWidget::PVOpcUaParamsWidget(PVInputTypeOpcUa const* in_t,
                                                 PVRush::hash_formats const& formats,
                                                 QWidget* parent)
    : PVParamsWidget<PVInputTypeOpcUa, PVOpcUaPresets, PVOpcUaInfos, PVOpcUaQuery>(
          in_t, formats, parent)
{
	_txt_host->setText("opc.tcp://your-opcua-server.com:4840/");
	_txt_host->setToolTip("opc.tcp://your-opcua-server.com:4840/");
	label_3->setVisible(false);
	_port_sb->setVisible(false);
	tabWidget->removeTab(1);

	_columns_tree_widget->header()->setSectionResizeMode(QHeaderView::ResizeToContents);

	_help_label->setText(R"html(
<html><head/><body>
	<p>Currently able to load integer nodes.</p>
	<p>Limitations of <a link="https://open62541.org">Open62541</a> implementation applies.</p>
</body></html>
		)html");
	
	tabWidget->setCurrentIndex(0);

	resize(QGuiApplication::screens()[0]->geometry().height() - 50, QGuiApplication::screens()[0]->geometry().width() / 2);
}

void PVRush::PVOpcUaParamsWidget::reset_columns_tree_widget()
{
	disconnect(_columns_tree_widget, &QTreeWidget::itemChanged, this,
	           &PVOpcUaParamsWidget::tree_item_changed);

	_columns_tree_widget->clear();
	_root_item = new QTreeWidgetItem(_columns_tree_widget);
	_root_item->setExpanded(true);
	_root_item->setText(0, "properties");
	_root_item->setCheckState(0, Qt::Unchecked);

	auto deserialized_query = get_serialize_query().split(QRegularExpression("\\;\\$\\;"));
	size_t nodes_count = deserialized_query.size() / 3;
	for (size_t i = 0; i < nodes_count; ++i) {
		// configure node per node
		auto column_node_id = deserialized_query[3 * i + 0];
		auto column_name = deserialized_query[3 * i + 1];
		auto column_type = deserialized_query[3 * i + 2];
		auto* tree_item = new QTreeWidgetItem(_root_item);
		tree_item->setText(0, "(" + column_node_id + ") " + column_name);
		tree_item->setData(0, Qt::UserRole, column_node_id);
		tree_item->setText(1, column_type);
		tree_item->setExpanded(true);
		tree_item->setCheckState(0, Qt::Checked);
	}

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

void PVRush::PVOpcUaParamsWidget::fetch_server_data_slot()
{
	fetch_server_data(get_infos());
}

//static bool connecting = false;

void PVRush::PVOpcUaParamsWidget::check_connection_slot()
{
	std::string error;

	if (not check_connection(&error)) {
		QMessageBox::critical(this, tr("Failure"), tr("Connection error : %1").arg(error.c_str()),
		                      QMessageBox::Ok);
	}
}

bool PVRush::PVOpcUaParamsWidget::check_connection(std::string* error /*= nullptr*/)
{
	const PVOpcUaInfos& infos = get_infos();

	QOpcUaProvider provider;
	if (provider.availableBackends().isEmpty()) {
		*error = "No OpcUa backend available!";
		qDebug() << error->c_str();
		return false;
	}
	QOpcUaClient* client = provider.createClient(provider.availableBackends()[0]);
	if (!client) {
		*error += "OpcUa backend (";
		*error += provider.availableBackends()[0].toStdString();
		*error += ") could not be loaded and could not create client.";
		qDebug() << error->c_str();
		return false;
	}


	QOpcUaPkiConfiguration pkiConfig;

#if 0 // Not yet supported
	QString pkidir = PVWidgets::PVFileDialog::getExistingDirectory();
	pkiConfig.setClientCertificateFile(PVWidgets::PVFileDialog::getOpenFileName());//certificate.der
	pkiConfig.setPrivateKeyFile(PVWidgets::PVFileDialog::getOpenFileName());//privatekey.pem
	pkiConfig.setTrustListDirectory(pkidir + "/trusted/certs");
	pkiConfig.setRevocationListDirectory(pkidir + "/trusted/crl");
	pkiConfig.setIssuerListDirectory(pkidir + "/issuers/certs");
	pkiConfig.setIssuerRevocationListDirectory(pkidir + "/issuers/crl");
#endif

	QOpcUaAuthenticationInformation authInfo;
	if (infos.get_login().isEmpty()) {
		authInfo.setAnonymousAuthentication();
	} else {
		authInfo.setUsernameAuthentication(infos.get_login(), infos.get_password());
	}

	client->setAuthenticationInformation(authInfo);
#if 0 // Not yet supported
	client->setPkiConfiguration(pkiConfig);
#endif
	client->setApplicationIdentity(pkiConfig.applicationIdentity());

	QObject::connect(
	    client, &QOpcUaClient::stateChanged, [client, this](QOpcUaClient::ClientState state) {
		    qDebug() << "Client state changed:" << state;
		    if (state == QOpcUaClient::ClientState::Connected) {
			    QOpcUaNode* node = client->node("ns=0;i=84");
				(void) node;
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
			    auto scroll_area = new QScrollArea;
				scroll_area->setWidget(_opcua_treeview);
				auto tab_index = tabWidget->addTab(scroll_area, "Browse nodes");
			    connect(_opcua_treeview->selectionModel(), &QItemSelectionModel::selectionChanged,
			            [this](const QItemSelection& /*selected*/, const QItemSelection& /*deselected*/) {
				            reset_columns_tree_widget();
//				            for (auto& selitem : selected) {
//					            //
//				            }
			            });
			    _opcua_treeview->setContextMenuPolicy(Qt::CustomContextMenu);
				connect(_opcua_treeview, &QTreeView::customContextMenuRequested, [this](QPoint const& pos){
					QModelIndex index = _opcua_treeview->indexAt(pos);
					if (index.isValid()) {
						auto contextMenu = new QMenu(this);
						contextMenu->addAction("Export node to CSV", [this, index](){
							export_node(index.siblingAtColumn(0).data(Qt::UserRole).value<QString>(),
							            index.siblingAtColumn(5).data(Qt::UserRole).value<QString>());
						});
						contextMenu->addAction("Export node to CSV (values only)", [this, index](){
							export_node(index.siblingAtColumn(0).data(Qt::UserRole).value<QString>(),
							            index.siblingAtColumn(5).data(Qt::UserRole).value<QString>(), false);
						});
						contextMenu->exec(_opcua_treeview->viewport()->mapToGlobal(pos));
					}
				});
				tabWidget->setCurrentIndex(tab_index);
				QMessageBox::information(this, tr("Success"), tr("Connection successful"), QMessageBox::Ok);
		    }
	    });

	connect(client, &QOpcUaClient::connectError, [this](QOpcUaErrorState* errorState) {
		std::string error = "Error State:" + QOpcUa::statusToString(errorState->errorCode()).toStdString();
		qDebug() << error.c_str();
		if (errorState->isClientSideError() &&
		    errorState->connectionStep() ==
		        QOpcUaErrorState::ConnectionStep::CertificateValidation) {
			errorState->setIgnoreError(true);
		} else {
			QMessageBox::critical(this, tr("Failure"), tr("Connection error : %1").arg(error.c_str()),
			                      QMessageBox::Ok);
		}
	});

	QObject::connect(client, &QOpcUaClient::endpointsRequestFinished,
	                 [client, this](QVector<QOpcUaEndpointDescription> endpoints,
	                          QOpcUa::UaStatusCode statusCode, QUrl requestUrl) {
		                 qDebug() << "Endpoints returned:" << endpoints.count() << statusCode
		                          << requestUrl;
		                 if (endpoints.size()) {
			                 endpoints.first().setEndpointUrl(
			                     requestUrl.toString()); // Needed to workaround an unconfigured
			                                             // reverse DNS.
			                 client->connectToEndpoint(
			                     endpoints.first()); // Connect to the first endpoint in the list
		                 } else {
			                 std::string error = "Endpoints Request Status:" + QOpcUa::statusToString(statusCode).toStdString();
			                 qDebug() << error.c_str();
			                 QMessageBox::critical(this, tr("Failure"), tr("Connection error : %1").arg(error.c_str()),
			                                       QMessageBox::Ok);
		                 }
	                 });

	if (client->requestEndpoints(QUrl(infos.get_host()))) {
		qDebug() << "OpcUa client requesting endpoints...";
		return true;
	} else {
		*error += "OpcUa client could not request endpoints.";
		qDebug() << error->c_str();
	}

	return false;
}

void PVRush::PVOpcUaParamsWidget::query_type_changed_slot()
{
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
	//bool query_end = false;

	PVRush::PVOpcUaAPI es(get_infos());
	//const PVOpcUaQuery& query = get_query(error);

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
			    column_node_id, UA_DateTime_fromUnixTime(0), UA_DateTime_now(), 10000,
			    [&compressor, &sep, &quote, &count](UA_HistoryData* data, bool) {
				    for (UA_UInt32 i = 0; i < data->dataValuesSize; ++i) {
					    UA_DataValue& value = data->dataValues[i];
					    std::string v;
					    v += safe_export(std::to_string(value.sourceTimestamp), sep, quote) + sep;
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

void PVRush::PVOpcUaParamsWidget::export_node(QString node_id, QString node_name, bool source_timestamp)
{
	std::string error;
	// FileDialog for option selection and file to write
	PVWidgets::PVExportDlg export_dlg;

	QFile file;
	QString filename;
	// Ask for file until a valid name is given or the action is aborted
	while (true) {
		int res = export_dlg.exec();
		filename = export_dlg.selectedFiles()[0];
		if (filename.isEmpty() || res == QDialog::Rejected) {
			return;
		}

		file.setFileName(filename);
		if (file.open(QIODevice::WriteOnly)) {

			break;
		}

		// Error case
		QMessageBox::critical(&export_dlg, tr("Error while exporting the selection"),
								tr("Can not create the file \"%1\"").arg(filename));
	}

	// Export query
	const PVRush::PVCSVExporter& exporter =
		dynamic_cast<PVRush::PVCSVExporter&>(export_dlg.exporter_widget()->exporter());
	std::string sep = exporter.get_sep_char(), quote = exporter.get_quote_char();
	bool header = exporter.get_export_header();
	PVCore::PVStreamingCompressor compressor(filename.toStdString());
	PVCore::PVProgressBox::progress(
		[&](PVCore::PVProgressBox& pbox) {
			size_t count = 0;

			PVRush::PVOpcUaAPI es(get_infos());

			using PVRush::PVUtils::safe_export;

			try {
				if (header) {
					if (source_timestamp) {
						compressor.write(safe_export("Source Timestamp", sep, quote) + sep +
										safe_export(node_name.toStdString().c_str(), sep, quote) + "\n");
					} else {
						compressor.write(safe_export(node_name.toStdString().c_str(), sep, quote) + "\n");
					}
				}

				es.read_node_history(
					node_id, UA_DateTime_fromUnixTime(0), UA_DateTime_now(), 10000,
					[&compressor, &sep, &quote, &count, &pbox, &error, source_timestamp](UA_HistoryData* data, bool) {
						if (not error.empty()) {
							return false;
						}
						if (pbox.get_cancel_state() == PVCore::PVProgressBox::CancelState::CANCEL ||
							pbox.get_cancel_state() == PVCore::PVProgressBox::CancelState::CANCEL2) {
							return false;
						}
						for (UA_UInt32 i = 0; i < data->dataValuesSize; ++i) {
							UA_DataValue& value = data->dataValues[i];
							auto str_value = PVOpcUaAPI::to_json_string(value.value);
							str_value = str_value.substr(1, str_value.size() - 2);
							if (source_timestamp) {
								std::string v;
								v += safe_export(std::to_string(value.sourceTimestamp), sep, quote) + sep;
								v +=
									safe_export(str_value, sep, quote) + "\n";
								compressor.write(v);
							} else {
								compressor.write(str_value + "\n");
							}
						}
						count += data->dataValuesSize;
						pbox.set_value(count);
						pbox.set_extended_status(QString("%L1 rows exported so far").arg(count));
						return true;
					});
			} catch (const PVCore::PVStreamingCompressorError& e) {
				error = e.what();
			}
		},
		"Exporting " + node_name + " (" + node_id + ")...", this);
	compressor.wait_finished();
}

bool PVRush::PVOpcUaParamsWidget::set_infos(PVOpcUaInfos const& infos)
{
	[[maybe_unused]] bool res =
	    PVParamsWidget<PVInputTypeOpcUa, PVOpcUaPresets, PVOpcUaInfos, PVOpcUaQuery>::set_infos(
	        infos);

	return true;
}

PVRush::PVOpcUaInfos PVRush::PVOpcUaParamsWidget::get_infos() const
{
	PVRush::PVOpcUaInfos infos =
	    PVParamsWidget<PVInputTypeOpcUa, PVOpcUaPresets, PVOpcUaInfos, PVOpcUaQuery>::get_infos();

	return infos;
}

QString PVRush::PVOpcUaParamsWidget::get_server_query(std::string* /*error  = nullptr */) const
{
	QString q = get_serialize_query();
	return q;
}

QString PVRush::PVOpcUaParamsWidget::get_serialize_query() const
{
	if (_opcua_treeview) {
		QString serialized;
		for (auto& opcua_column : _opcua_treeview->selectionModel()->selectedRows()) {
			auto column_node_id = opcua_column.siblingAtColumn(0).data(Qt::UserRole).value<QString>();
			auto column_type = opcua_column.siblingAtColumn(3).data(Qt::UserRole).value<QString>();
			auto column_name = opcua_column.siblingAtColumn(5).data(Qt::UserRole).value<QString>();
			serialized += column_node_id + ";$;" + column_type + ";$;" + column_name + ";$;";
		}
		qDebug() << __func__ << serialized;
		return serialized;
	} else {
		qDebug() << __func__ << _serialized_query;
		return _serialized_query;
	}
}

void PVRush::PVOpcUaParamsWidget::set_query(QString const& query)
{
	_serialized_query = query;
	reset_columns_tree_widget();
	qDebug() << __func__ << query;
}

size_t PVRush::PVOpcUaParamsWidget::query_result_count(std::string* /*error  = nullptr */)
{
	return 0;
}

bool PVRush::PVOpcUaParamsWidget::fetch_server_data(const PVOpcUaInfos& /*infos*/)
{
	return true;
}

size_t PVRush::PVOpcUaParamsWidget::get_selected_columns_count() const
{
	if (_opcua_treeview) {
		return _opcua_treeview->selectionModel()->selectedRows().size();
	} else {
		return _serialized_query.split(QRegularExpression("\\;\\$\\;")).size() / 3;
	}
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
	if (1 + fields_count < 2) {
		QMessageBox::critical(this, "Invalid format error",
		                      "At least two columns must be selected to have a valid format.");
		return;
	}

	if (is_format_custom()) {
		update_custom_format();
	} else if (1 + fields_count != get_first_level_fields_count(_format_path->text())) {
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

	set_query(get_serialize_query());

	auto deserialized_query = _serialized_query.split(QRegularExpression("\\;\\$\\;"));
	size_t nodes_count = deserialized_query.size() / 3;
	for (size_t i = 0; i < nodes_count; ++i) {
		// configure node per node
		auto column_name = deserialized_query[3 * i + 2];
		auto node_id_open62541 = PVOpcUaAPI::NodeId(deserialized_query[3 * i + 1]).open62541();
		if (auto* data_type = UA_findDataType(&node_id_open62541)) {
			/*PVRush::PVXmlTreeNodeDom* node =*/
				format_root->addOneField(column_name, QString(PVRush::PVOpcUaAPI::pvcop_type(data_type->typeIndex)));
		} else {
			/*PVRush::PVXmlTreeNodeDom* node = */format_root->addOneField(column_name, QString("uint8"));
		}
	}
}

void PVRush::PVOpcUaParamsWidget::edit_custom_format()
{
	update_custom_format();
	PVParamsWidgetBase::edit_custom_format();
}
