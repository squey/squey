#include <openssl/ssl.h>
#include <openssl/opensslv.h>

#include <QSslSocket>
#include <QFile>
#include <QDir>

#include <iostream>
#include <memory>

#include <QOpcUaClient>
#include <QOpcUaProvider>
#include <QOpcUaPkiConfiguration>
#include <QOpcUaAuthenticationInformation>
#include <QOpcUaErrorState>

#include <QApplication>
#include <QWidget>
#include <QTreeView>
#include <QHeaderView>

#include <open62541.h>

#include "../../plugins/common/opcua/PVOpcUaTreeModel.h"
#include "../../plugins/common/opcua/PVOpcUaQuery.h"

#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVCSVExporter.h>

#include <pvkernel/core/inendi_assert.h>
#include "common.h"

static char* PKI_DIR = nullptr;
static char* SERVER_URL = nullptr;
static char* SERVER_USER = nullptr;
static char* SERVER_PASSWORD = nullptr;

#define UATOQSTRING(uastr)                                                                         \
	QString::fromUtf8(reinterpret_cast<const char*>(uastr.data), uastr.length)
#define QDBGVAR(part) #part "=" << part
#define QDBGSTR(part) #part "=" << UATOQSTRING(part)
#define QDBGSTS(part)                                                                              \
#part "=" << (part == 0x80020000 ? "UA_STATUSCODE_BADINTERNALERROR"                            \
	                                 : part == 0x0 ? "UA_STATUSCODE_GOOD" : "UA_STATUSCODE_OTHER") \
	          << part
#define QDBGF_START qDebug() << "Start:" << __func__;
#define QDBGF_END qDebug() << "End:" << __func__;

class SSLWidget : public QWidget
{
  public:
	SSLWidget()
	    : m_tree_view(new QTreeView(this)), m_opcua_model(new PVRush::PVOpcUaTreeModel(this))
	{
		m_tree_view->setModel(m_opcua_model);
		m_tree_view->header()->setSectionResizeMode(QHeaderView::ResizeToContents);
		m_tree_view->setTextElideMode(Qt::ElideRight);
		m_tree_view->setAlternatingRowColors(true);
		m_tree_view->setSelectionBehavior(QAbstractItemView::SelectItems);
	}

	void resizeEvent(QResizeEvent*) override
	{
		m_tree_view->setMinimumWidth(width());
		m_tree_view->setMinimumHeight(height());
	}

	bool opcua();

  private:
	QTreeView* m_tree_view;
	PVRush::PVOpcUaTreeModel* m_opcua_model;
	std::unique_ptr<QOpcUaClient> m_opcua_client = nullptr;
};

static auto pki_config()
{
	QString pkidir(PKI_DIR);
	QOpcUaPkiConfiguration pkiConfig;
	pkiConfig.setClientCertificateFile(pkidir + "/own/certs/lay2form_fchapelle_certificate.der");
	pkiConfig.setPrivateKeyFile(pkidir + "/own/private/lay2form_fchapelle_privatekey.pem");
	pkiConfig.setTrustListDirectory(pkidir + "/trusted/certs");
	pkiConfig.setRevocationListDirectory(pkidir + "/trusted/crl");
	pkiConfig.setIssuerListDirectory(pkidir + "/issuers/certs");
	pkiConfig.setIssuerRevocationListDirectory(pkidir + "/issuers/crl");
	return pkiConfig;
}

static std::unique_ptr<QOpcUaClient> create_client()
{
	QOpcUaProvider provider;
	if (provider.availableBackends().isEmpty()) {
		qDebug() << "No OpcUa backend available!";
		return nullptr;
	}
	std::unique_ptr<QOpcUaClient> client{provider.createClient(provider.availableBackends()[0])};
	if (!client) {
		qDebug() << "OpcUa backend (" << provider.availableBackends()[0]
		         << ") could not be loaded and could not create client.";
		return nullptr;
	}
	qDebug() << "supportedSecurityPolicies()" << client->supportedSecurityPolicies();
	qDebug() << "supportedUserTokenTypes()" << client->supportedUserTokenTypes();
	return client;
}

bool SSLWidget::opcua()
{
	QDBGF_START
	m_opcua_client = create_client();
	if (!m_opcua_client) {
		return false;
	}

	QOpcUaPkiConfiguration pkiConfig = pki_config();

	QOpcUaAuthenticationInformation authInfo;
	// authInfo.setCertificateAuthentication();
	// authInfo.setAnonymousAuthentication();
	authInfo.setUsernameAuthentication(SERVER_USER, SERVER_PASSWORD);

	m_opcua_client->setAuthenticationInformation(authInfo);
	m_opcua_client->setPkiConfiguration(pkiConfig);
	m_opcua_client->setApplicationIdentity(pkiConfig.applicationIdentity());

	connect(m_opcua_client.get(), &QOpcUaClient::stateChanged,
	        [this](QOpcUaClient::ClientState state) {
		        qDebug() << "Client state changed:" << state;
		        if (state == QOpcUaClient::ClientState::Connected) {
			        QOpcUaNode* node = m_opcua_client->node("ns=0;i=84");
			        if (node)
				        qDebug() << "A node object has been created";
			        m_opcua_model->setOpcUaClient(m_opcua_client.get());
			        m_tree_view->header()->setSectionResizeMode(1 /* Value column*/,
			                                                    QHeaderView::Interactive);
		        }
	        });

	connect(m_opcua_client.get(), &QOpcUaClient::connectError, [](QOpcUaErrorState* errorState) {
		qDebug() << "Client Error State:" << QOpcUa::statusToString(errorState->errorCode());
		if (errorState->isClientSideError() &&
		    errorState->connectionStep() ==
		        QOpcUaErrorState::ConnectionStep::CertificateValidation) {
			errorState->setIgnoreError(true);
		}
	});

	connect(m_opcua_client.get(), &QOpcUaClient::endpointsRequestFinished,
	        [this](QVector<QOpcUaEndpointDescription> endpoints, QOpcUa::UaStatusCode statusCode,
	               QUrl requestUrl) {
		        qDebug() << "Endpoints returned:" << endpoints.count() << statusCode << requestUrl;

		        if (endpoints.size()) {
			        auto& endpoint = endpoints.first();
			        qDebug() << "Endpoint[0]:" << QDBGVAR(endpoint.endpointUrl())
			                 << QDBGVAR(endpoint.securityMode())
			                 << QDBGVAR(endpoint.securityPolicy())
			                 << QDBGVAR(endpoint.transportProfileUri())
			                 << QDBGVAR(endpoint.securityLevel()) << "endpoint.userIdentityTokens:";
			        for (auto const& uit : endpoint.userIdentityTokens()) {
				        qDebug() << QDBGVAR(uit.policyId()) << QDBGVAR(uit.tokenType())
				                 << QDBGVAR(uit.issuedTokenType())
				                 << QDBGVAR(uit.issuerEndpointUrl())
				                 << QDBGVAR(uit.securityPolicy());
			        }

			        endpoint.setEndpointUrl(
			            requestUrl.toString()); // Needed to workaround an unconfigured reverse DNS.
			        m_opcua_client->connectToEndpoint(endpoint);
		        }
	        });

	connect(
	    m_opcua_client.get(), &QOpcUaClient::findServersFinished,
	    [&](QVector<QOpcUaApplicationDescription> servers, QOpcUa::UaStatusCode statusCode,
	        QUrl requestUrl) {
		    qDebug() << "Application descriptions returned: " << servers.size() << statusCode
		             << requestUrl;
		    if (statusCode == QOpcUa::Good) {
			    // Choose application description from vector
			    QOpcUaApplicationDescription applicationDescription = servers[0];
			    if (m_opcua_client->requestEndpoints(applicationDescription.discoveryUrls()[0])) {
				    qDebug() << "OpcUa client requesting endpoints...";
			    } else {
				    qDebug() << "OpcUa client could not request endpoints.";
			    }
		    }
	    });

	if (m_opcua_client->requestEndpoints(QUrl(SERVER_URL))) {
		qDebug() << "OpcUa client requesting endpoints...";
	} else {
		qDebug() << "OpcUa client could not request endpoints.";
	}

	// if (client->findServers(QUrl(SERVER_URL))) {
	// 	qDebug() << "OpcUa client finding servers...";
	// }
	// else { qDebug() << "OpcUa client could not search servers."; }
	QDBGF_END
	return true;
}

bool loadFileToByteString(const QString& location, UA_ByteString* target)
{
	if (location.isEmpty()) {
		qDebug() << "Unable to read from empty file path";
		return false;
	}

	if (!target) {
		qDebug() << "No target ByteString given";
		return false;
	}

	UA_ByteString_init(target);

	QFile file(location);

	if (!file.open(QFile::ReadOnly)) {
		qWarning() << "Failed to open file" << location;
		return false;
	}

	QByteArray data = file.readAll();

	UA_ByteString temp;
	temp.length = data.length();
	if (data.isEmpty())
		temp.data = nullptr;
	else {
		if (location.endsWith(".pem")) {
			// Remove trailing newline
			data = QString::fromLatin1(data).trimmed().toLatin1();
		}
		temp.data = reinterpret_cast<unsigned char*>(data.data());
	}

	bool success = UA_ByteString_copy(&temp, target);

	return success == UA_STATUSCODE_GOOD;
}

bool loadAllFilesInDirectory(const QString& location, UA_ByteString** target, int* size)
{
	if (location.isEmpty()) {
		qDebug() << "Unable to read from empty file path";
		return false;
	}

	if (!target) {
		qDebug() << "No target ByteString given";
		return false;
	}

	*target = nullptr;
	*size = 0;

	QDir dir(location);

	auto entries = dir.entryList(QDir::Files);

	if (entries.isEmpty()) {
		qDebug() << "Directory is empty";
		return true;
	}

	int tempSize = entries.size();
	UA_ByteString* list =
	    static_cast<UA_ByteString*>(UA_Array_new(tempSize, &UA_TYPES[UA_TYPES_BYTESTRING]));

	if (!list) {
		qDebug() << "Failed to allocate memory for loading files in" << location;
		return false;
	}

	for (int i = 0; i < entries.size(); ++i) {
		if (!loadFileToByteString(dir.path() + QChar('/') + entries.at(i), &list[i])) {
			qWarning() << "Failed to open file" << entries.at(i);
			UA_Array_delete(list, tempSize, &UA_TYPES[UA_TYPES_BYTESTRING]);
			size = 0;
			*target = nullptr;
			return false;
		}
	}

	*target = list;
	*size = tempSize;

	return true;
}

void basic_openssl()
{
	SSL_load_error_strings();
	ERR_load_BIO_strings();
	OpenSSL_add_all_algorithms();
	OpenSSL_add_all_digests();

	std::cout << "OpenSSL " << SSLeay_version(SSLEAY_VERSION) << std::endl;

	qDebug() << "sslLibraryVersionString=" << QSslSocket::sslLibraryVersionString();
	qDebug() << "supportsSsl=" << QSslSocket::supportsSsl();
}

void request_endpoints()
{
	QDBGF_START
	UA_Client* client = UA_Client_new();
	UA_ClientConfig_setDefault(UA_Client_getConfig(client));
	size_t numEndpoints = 0;
	UA_EndpointDescription* endpoints = nullptr;
	UA_StatusCode retval = UA_Client_getEndpoints(client, SERVER_URL, &numEndpoints, &endpoints);

	qDebug() << "UA_Client_getEndpoints:" << QDBGSTS(retval) << QDBGVAR(numEndpoints);

	qDebug() << "endpoints[0]:" << QDBGSTR(endpoints[0].endpointUrl)
	         << QDBGVAR(endpoints[0].securityMode) << QDBGSTR(endpoints[0].securityPolicyUri)
	         << QDBGSTR(endpoints[0].transportProfileUri) << QDBGVAR(endpoints[0].securityLevel);

	for (size_t i = 0; i < endpoints[0].userIdentityTokensSize; ++i) {
		auto& token = endpoints[0].userIdentityTokens[i];
		qDebug() << "endpoints[0].userIdentityTokens[" << '0' + i << "]:" << QDBGSTR(token.policyId)
		         << QDBGVAR(token.tokenType) << QDBGSTR(token.issuedTokenType)
		         << QDBGSTR(token.issuerEndpointUrl) << QDBGSTR(token.securityPolicyUri);
	}

	UA_Client_disconnect(client);
	UA_Client_delete(client);
	QDBGF_END
}

static void printTimestamp(char const* name, UA_DateTime date)
{
	UA_DateTimeStruct dts = UA_DateTime_toStruct(date);
	if (name)
		printf("%s: %02u-%02u-%04u %02u:%02u:%02u.%03u, ", name, dts.day, dts.month, dts.year,
		       dts.hour, dts.min, dts.sec, dts.milliSec);
	else
		printf("%02u-%02u-%04u %02u:%02u:%02u.%03u, ", dts.day, dts.month, dts.year, dts.hour,
		       dts.min, dts.sec, dts.milliSec);
}

static void printDataValue(UA_DataValue* value)
{
	/* Print status and timestamps */
	if (value->hasServerTimestamp)
		printTimestamp("ServerTime", value->serverTimestamp);

	if (value->hasSourceTimestamp)
		// printTimestamp("SourceTime", value->sourceTimestamp);
		printf("SourceTime %ld, ", value->sourceTimestamp);

	if (value->hasStatus)
		printf("Status 0x%08x, ", value->status);

	if (value->value.type == &UA_TYPES[UA_TYPES_UINT32]) {
		UA_UInt32 hrValue = *(UA_UInt32*)value->value.data;
		printf("Uint32Value %u\n", hrValue);
	}

	if (value->value.type == &UA_TYPES[UA_TYPES_DOUBLE]) {
		UA_Double hrValue = *(UA_Double*)value->value.data;
		printf("DoubleValue %f\n", hrValue);
	}

	if (value->value.type == &UA_TYPES[UA_TYPES_INT64]) {
		UA_Int64 hrValue = *(UA_Int64*)value->value.data;
		printf("Int64Value %ld\n", hrValue);
	}

	if (value->value.type == &UA_TYPES[UA_TYPES_STRING]) {
		UA_String hrValue = *(UA_String*)value->value.data;
		printf("StringValue %s\n", hrValue.data);
	}

	// if (value->value.type) {
	// 	qDebug() << "ValueType:" << value->value.type->typeName;
	// }
}

static UA_Boolean readRaw(const UA_HistoryData* data)
{
	printf("readRaw Value count: %lu\n", (long unsigned)data->dataValuesSize);
	// printDataValue(&data->dataValues[0]);

	/* Iterate over all values */
	for (UA_UInt32 i = 0; i < data->dataValuesSize; ++i) {
		printDataValue(&data->dataValues[i]);
	}

	/* We want more data! */
	return true;
}

static UA_Boolean readHist(UA_Client* client,
                           const UA_NodeId* nodeId,
                           UA_Boolean moreDataAvailable,
                           const UA_ExtensionObject* data,
                           void* unused)
{
	qDebug() << "Read historical callback (Has more data:" << moreDataAvailable << "):";
	if (data->content.decoded.type == &UA_TYPES[UA_TYPES_HISTORYDATA]) {
		return readRaw((UA_HistoryData*)data->content.decoded.data);
	}
	return true;
}

void connect_to_server()
{
	QDBGF_START
	UA_Client* client = UA_Client_new();
	UA_ClientConfig* conf = UA_Client_getConfig(client);

	{
		auto pki = pki_config();
		UA_ByteString localCertificate;
		UA_ByteString privateKey;
		UA_ByteString* trustList = nullptr;
		int trustListSize = 0;
		UA_ByteString* revocationList = nullptr;
		int revocationListSize = 0;

		if (loadFileToByteString(pki.clientCertificateFile(), &localCertificate) and
		    loadFileToByteString(pki.privateKeyFile(), &privateKey) and
		    loadAllFilesInDirectory(pki.trustListDirectory(), &trustList, &trustListSize) and
		    loadAllFilesInDirectory(pki.revocationListDirectory(), &revocationList,
		                            &revocationListSize)) {
			UA_StatusCode retval = UA_ClientConfig_setDefaultEncryption(
			    conf, localCertificate, privateKey, trustList, trustListSize, revocationList,
			    revocationListSize);
			qDebug() << "UA_ClientConfig_setDefaultEncryption():" << QDBGSTS(retval);
		} else {
			qDebug() << "Failed to load conf, aborting connection process.";
		}
	}

	UA_StatusCode retval =
	    UA_Client_connect_username(client, SERVER_URL, SERVER_USER, SERVER_PASSWORD);

	qDebug() << "UA_Client_connect_username:" << QDBGSTS(retval);

	/* Read historical values (uint32) */
	printf("\nStart historical read (1, \"myUintValue\"):\n");
	// UA_NodeId node = UA_NODEID_STRING(2, "MyLevel");
	UA_NodeId node = UA_NODEID_NUMERIC(2, 2021);
	retval = UA_Client_HistoryRead_raw(client, &node, readHist, UA_DateTime_fromUnixTime(0),
	                                   UA_DateTime_now(), UA_STRING_NULL, false, 100,
	                                   UA_TIMESTAMPSTORETURN_BOTH, (void*)UA_FALSE);

	// retval = UA_Client_HistoryRead_atTime(
	//     client, &node, readHist, UA_DateTime_fromUnixTime(132113687876623850),
	//     UA_STRING_NULL, false, UA_TIMESTAMPSTORETURN_BOTH, (void*)UA_FALSE);

	if (retval != UA_STATUSCODE_GOOD) {
		printf("Failed. %s\n", UA_StatusCode_name(retval));
	}

	UA_Client_disconnect(client);
	UA_Client_delete(client);
	QDBGF_END
}

void qtopcua_connect(int argc, char** argv)
{
	QApplication app(argc, argv);
	SSLWidget widget;
	widget.opcua();
	widget.showMaximized();
	app.exec();
}

int test_source(int argc, char** argv)
{
	std::string const& format_file = TEST_FOLDER "/pvkernel/rush/opcua/opcua-testsuite.format";

	pvtest::init_ctxt();

	PVRush::PVOpcUaInfos infos;
	infos.set_host("opc.tcp://opcua-testsuite.srv.picviz:4850/");

	std::string query_str =
	"ns=2;i=2011;$;ns=0;i=11;$;P6.1.1 Temperature;$;"
	"ns=2;i=2015;$;ns=0;i=11;$;P6.2.1 FastSpeedDownward;$;"
	"ns=2;i=2017;$;ns=0;i=11;$;P6.2.2 SlowSpeedDownward;$;"
	"ns=2;i=2018;$;ns=0;i=11;$;P6.2.3 SlowSpeedUpward;$;"
	"ns=2;i=2016;$;ns=0;i=11;$;P6.2.4 FastSpeedUpward;$;"
	"ns=2;i=2019;$;ns=0;i=11;$;P6.2.5 FastSlowPosition;$;"
	"ns=2;i=2020;$;ns=0;i=11;$;6.2.6 SlowFastPosition;$;";

	const size_t expected_row_count = 193'953;

	PVRush::PVFormat format("format", QString::fromStdString(format_file));

	/**************************************************************************
	 * Import data
	 *************************************************************************/
	PVRush::PVSourceCreator_p sc =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("opcua");

	QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
	auto opcua_desc =
		new PVRush::PVOpcUaQuery(infos, QString::fromStdString(query_str), QString());
	list_inputs << PVRush::PVInputDescription_p(opcua_desc);

	PVRush::PVNraw nraw;
	PVRush::PVNrawOutput output(nraw);
	PVRush::PVExtractor extractor(format, output, sc, list_inputs);

	// Import data
	auto start = std::chrono::system_clock::now();
	PVRush::PVControllerJob_p job =
		extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT);
	job->wait_end();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifndef INSPECTOR_BENCH
	PV_ASSERT_VALID((nraw.row_count() == expected_row_count));
#endif

#if 0 // Test is too big to store ref_file in tests repository
	// Export selected lines
	PVCore::PVSelBitField sel(nraw.row_count());
	sel.select_all();
	const std::string& output_file = pvtest::get_tmp_filename();
	PVRush::PVCSVExporter::export_func_f export_func =
		[&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
			const std::string& quote) { return nraw.export_line(row, cols, sep, quote); };
	PVRush::PVCSVExporter exp(format.get_axes_comb(), nraw.row_count(), export_func);

	exp.export_rows(output_file, sel);

#define INSPECTOR_BENCH
#ifndef INSPECTOR_BENCH
	// Check output is the same as the reference
	std::cout << std::endl << output_file << " - " << ref_file << std::endl;
	PV_ASSERT_VALID(
		PVRush::PVUtils::files_have_same_content(output_file, ref_file));
#endif

	std::remove(output_file.c_str());
#endif

	return 0;
}


int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
	//return EXIT_SUCCESS; // Bypass test for current prod, FIXME
	return test_source(argc, argv);

	// if (!(PKI_DIR = getenv("OPCUA_PKIDIR")) or !(SERVER_URL = getenv("OPCUA_URL")) or
	//     !(SERVER_USER = getenv("OPCUA_USER")) or !(SERVER_PASSWORD = getenv("OPCUA_PASSWORD"))) {
	// 	qDebug() << "Please define env variables OPCUA_PKIDIR OPCUA_URL OPCUA_USER OPCUA_PASSWORD";
	// 	return EXIT_FAILURE;
	// }

	// request_endpoints();
	// connect_to_server();

	// qtopcua_connect(argc, argv);
}