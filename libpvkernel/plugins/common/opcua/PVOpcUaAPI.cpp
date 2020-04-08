/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#include "PVOpcUaAPI.h"

#include "PVOpcUaInfos.h"

#include <system_error>

#include <QOpcUaPkiConfiguration>
#include <QDebug>
#include <QRegExp>

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

std::error_category const& opcua_category()
{
	static struct opcua_error_category: public std::error_category
	{
		const char* name() const noexcept override { return "open62541"; }
		std::string message(int condition) const override {
			return UA_StatusCode_name(static_cast<UA_StatusCode>(condition));
		}
	} category;

	return category;
}

std::error_code make_opcua_error_code(UA_StatusCode status)
{
	return std::error_code(static_cast<int>(status), opcua_category());
}

template <class... Args>
std::system_error make_opcua_error(UA_StatusCode status, Args&&... args)
{
	return std::system_error(make_opcua_error_code(status), std::forward<Args>(args)...);
}

template <class... Args>
void check_opcua_code(UA_StatusCode status, Args&&... args)
{
	if (status != UA_STATUSCODE_GOOD){
		throw make_opcua_error(status, std::forward<Args>(args)...);
	}
}

static const std::unordered_map<int, const char*> opcua_to_pvcop_type{
    {UA_TYPES_BOOLEAN, "number_uint8"}, {UA_TYPES_SBYTE, "number_int8"},
    {UA_TYPES_BYTE, "number_uint8"},    {UA_TYPES_INT16, "number_int16"},
    {UA_TYPES_UINT16, "number_uint16"}, {UA_TYPES_INT32, "number_int32"},
    {UA_TYPES_UINT32, "number_uint32"}, {UA_TYPES_INT64, "number_int64"},
    {UA_TYPES_UINT64, "number_uint64"}, {UA_TYPES_FLOAT, "number_float"},
    {UA_TYPES_DOUBLE, "number_double"}, {UA_TYPES_STRING, "string"},
    {UA_TYPES_DATETIME, "datetime_us"}, {UA_TYPES_GUID, "string"},
    {UA_TYPES_BYTESTRING, "string"}};

PVRush::PVOpcUaAPI::PVOpcUaAPI(PVOpcUaInfos const& infos) : _infos(infos)
{
	connect_to_server();
}

PVRush::PVOpcUaAPI::~PVOpcUaAPI()
{
	disconnect();
}

void PVRush::PVOpcUaAPI::connect_to_server()
{
	disconnect();
	_client = UA_Client_new();

	setup_encryption();

	auto endpoint_url = _infos.get_host().toUtf8().data();
	if (_infos.get_login().isEmpty()) {
		UA_StatusCode retval = UA_Client_connect(_client, endpoint_url);
		qDebug() << "UA_Client_connect:" << QDBGSTS(retval);
		check_opcua_code(retval);
	} else {
		auto login = _infos.get_login().toUtf8();
		auto password = _infos.get_password().toUtf8();
		UA_StatusCode retval = UA_Client_connect_username(_client, endpoint_url, login.data(), password.data());
		qDebug() << "UA_Client_connect_username:" << QDBGSTS(retval);
		check_opcua_code(retval);
	}
}

void PVRush::PVOpcUaAPI::disconnect()
{
	if (_client) {
		UA_Client_disconnect(_client);
		UA_Client_delete(_client);
		_client = nullptr;
	}
}

static auto pki_config()
{
	QString pkidir("/home/fchapelle/dev/qtopcua/lay2form/pkidir");
	QOpcUaPkiConfiguration pkiConfig;
	pkiConfig.setClientCertificateFile(pkidir + "/own/certs/lay2form_fchapelle_certificate.der");
	pkiConfig.setPrivateKeyFile(pkidir + "/own/private/lay2form_fchapelle_privatekey.pem");
	pkiConfig.setTrustListDirectory(pkidir + "/trusted/certs");
	pkiConfig.setRevocationListDirectory(pkidir + "/trusted/crl");
	pkiConfig.setIssuerListDirectory(pkidir + "/issuers/certs");
	pkiConfig.setIssuerRevocationListDirectory(pkidir + "/issuers/crl");
	return pkiConfig;
}

static bool loadFileToByteString(const QString& location, UA_ByteString* target)
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

static bool loadAllFilesInDirectory(const QString& location, UA_ByteString** target, int* size)
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

void PVRush::PVOpcUaAPI::setup_encryption()
{
	UA_ClientConfig* conf = UA_Client_getConfig(_client);
	if (false) {
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
	} else {
		UA_ClientConfig_setDefault(conf);
	}
}

static UA_Boolean read_node_history_static(UA_Client* client,
                                           const UA_NodeId* nodeId,
                                           UA_Boolean moreDataAvailable,
                                           const UA_ExtensionObject* data,
                                           void* context)
{
	return static_cast<PVRush::PVOpcUaAPI*>(context)->read_history_data(nodeId, moreDataAvailable, data);
}

void PVRush::PVOpcUaAPI::read_node_history(NodeId node_id, UA_DateTime start_time, UA_DateTime end_time, uint32_t values_per_node, std::function<bool(UA_HistoryData*, bool)> callback)
{
	_read_callback = callback;
	UA_NodeId node = node_id.open62541();
	UA_StatusCode retval = UA_Client_HistoryRead_raw(
	    _client, &node, read_node_history_static, start_time, end_time, UA_STRING_NULL, false,
	    values_per_node, UA_TIMESTAMPSTORETURN_SOURCE, (void*)this);
	// UA_NodeId aggregate_type = UA_NODEID_STRING_ALLOC(0, "Count");
	// UA_NodeId aggregate_type = UA_NODEID_NUMERIC(0, 2352);
	// UA_StatusCode retval = UA_Client_HistoryRead_processed(
	//     _client, &node, read_node_history_static, UA_DateTime_now() - 100*UA_DATETIME_SEC,
	//     UA_DateTime_now(), 0, &aggregate_type, UA_STRING_NULL, UA_TIMESTAMPSTORETURN_BOTH,
	//     (void*)this);
	check_opcua_code(retval);
}

UA_DateTime PVRush::PVOpcUaAPI::first_historical_datetime(NodeId node_id)
{
	UA_DateTime ret_val = 0;
	_read_callback = [&ret_val](UA_HistoryData* data, bool) {
		if (data->dataValuesSize > 0) {
			ret_val = data->dataValues[0].sourceTimestamp;
		}
		return false;
	};
	UA_NodeId node = node_id.open62541();
	check_opcua_code(UA_Client_HistoryRead_raw(
	    _client, &node, read_node_history_static, UA_DateTime(1*UA_DATETIME_SEC), UA_DateTime(0),
	    UA_STRING_NULL, false, 1, UA_TIMESTAMPSTORETURN_BOTH, (void*)this));
	return ret_val;
}

bool PVRush::PVOpcUaAPI::read_history_data(const UA_NodeId* nodeId,
                                           UA_Boolean moreDataAvailable,
                                           const UA_ExtensionObject* data)
{
	if (data->content.decoded.type == &UA_TYPES[UA_TYPES_HISTORYDATA]) {
		return _read_callback((UA_HistoryData*)data->content.decoded.data, moreDataAvailable);
	}
	return true;
}

PVRush::PVOpcUaAPI::NodeId::NodeId(QString id)
{
	QRegExp rx("ns=(\\d+)\\;(i|s|b|g)=(.*)");
	rx.indexIn(id);
	qDebug() << rx.capturedTexts();
	bool ok = true;
	auto ns = rx.cap(1).toUShort(&ok);
	auto id_type = rx.cap(2);
	if (id_type == "i") {
		auto id_numeric = rx.cap(3).toUInt(&ok);
		_node_id = UA_NODEID_NUMERIC(ns, id_numeric);
	} else if (id_type == "s") {
		auto id_string = rx.cap(3).toUtf8();
		_node_id = UA_NODEID_STRING_ALLOC(ns, id_string.constData());
	} else if (id_type == "g") {
		qDebug() << "Unimplemented GUID UA_NodeId";
		_node_id = UA_NODEID_NULL;
	} else if (id_type == "b") {
		auto id_string = rx.cap(3).toUtf8();
		_node_id = UA_NODEID_BYTESTRING_ALLOC(ns, id_string.constData());
	} else {
		qDebug() << "Unimplemented unknown UA_NodeId with type " << id_type;
		_node_id = UA_NODEID_NULL;
	}
	assert(ok);
}

PVRush::PVOpcUaAPI::NodeId::~NodeId()
{
	UA_NodeId_deleteMembers(&_node_id);
}

const char* PVRush::PVOpcUaAPI::pvcop_type(int opcua_type_index)
{
	if (auto it = opcua_to_pvcop_type.find(opcua_type_index); it != opcua_to_pvcop_type.end()) {
		return it->second;
	} else {
		return "string";
	}
}

std::string PVRush::PVOpcUaAPI::to_json_string(UA_Variant const& value)
{
	std::string json_value;
	if (value.arrayLength) {
		json_value.resize(UA_calcSizeJson(&value, &UA_TYPES[UA_TYPES_VARIANT], nullptr, 0,
		                                  nullptr, 0, true));
		auto* json_value_pos = json_value.data();
		auto* json_value_end = json_value.data() + json_value.size();
		UA_encodeJson(&value, &UA_TYPES[UA_TYPES_VARIANT], (uint8_t**)&json_value_pos,
		              (const uint8_t**)&json_value_end, nullptr, 0, nullptr, 0, true);
	} else {
		json_value.resize(
		    UA_calcSizeJson(value.data, value.type, nullptr, 0, nullptr, 0, true));
		auto* json_value_pos = json_value.data();
		auto* json_value_end = json_value.data() + json_value.size();
		UA_encodeJson(value.data, value.type, (uint8_t**)&json_value_pos,
		              (const uint8_t**)&json_value_end, nullptr, 0, nullptr, 0, true);
	}
	return json_value;
}

void PVRush::PVOpcUaAPI::print_datetime(UA_DateTime date)
{
	UA_DateTimeStruct dts = UA_DateTime_toStruct(date);
	printf("%02u-%02u-%04u %02u:%02u:%02u.%03u\n", dts.day, dts.month, dts.year, dts.hour, dts.min,
	       dts.sec, dts.milliSec);
}

boost::posix_time::ptime PVRush::PVOpcUaAPI::to_ptime(UA_DateTime date)
{
	static const UA_DateTimeStruct time_zero_dts = UA_DateTime_toStruct(0);
	static const boost::posix_time::ptime time_zero(
	    boost::gregorian::date(time_zero_dts.year, time_zero_dts.month, time_zero_dts.day));

	return time_zero + boost::posix_time::microsec(date / UA_DATETIME_USEC);
}
