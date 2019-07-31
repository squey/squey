/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#include "PVOpcUaSource.h"

#include <QDebug>
#include <stdexcept>
#include <open62541.h>
#include <QOpcUaPkiConfiguration>
#include <QRegularExpression>

#include "../../common/opcua/PVOpcUaQuery.h"

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

PVRush::PVOpcUaSource::PVOpcUaSource(PVRush::PVInputDescription_p input)
    : _query(dynamic_cast<PVOpcUaQuery&>(*input))
{
	qDebug() << "PVOpcUaSource::operator()";
	qDebug() << _query.human_name() << _query.get_query() << _query.get_query_type();

	qDebug() << "Infos:" << _query.infos().get_host();

	auto serialized_query = _query.get_query();
	auto deserialized_query = serialized_query.split(QRegularExpression("\\;\\$\\;"));
	qDebug() << deserialized_query;
	_nodes_count = deserialized_query.size() / 3;
	for (size_t i = 0; i < _nodes_count; ++i) {
		// configure node per node
	}
	// connect_to_server();
}

QString PVRush::PVOpcUaSource::human_name()
{
	return QString("opcua");
}

void PVRush::PVOpcUaSource::seek_begin() {}

void PVRush::PVOpcUaSource::prepare_for_nelts(chunk_index /*nelts*/) {}

size_t PVRush::PVOpcUaSource::get_size() const
{
	return 0;
}

PVCore::PVBinaryChunk* PVRush::PVOpcUaSource::operator()()
{
	if (_current_chunk != 0) {
		return nullptr;
	}

	auto chsize = 1000;

	PVCore::PVBinaryChunk& chunk = *_chunks.emplace_back(std::make_unique<PVCore::PVBinaryChunk>(_nodes_count, chsize, 0));

	_data.resize(chsize, 42);
	for (int i = 0; i < chsize; ++i) {
		_data[i] = i;
	}
	for (int i = 0; i < _nodes_count; ++i) {
		chunk.set_column_chunk(PVCol(i), _data);
	}
	chunk.set_rows_count(_data.size());
	qDebug() << "chunk filled";
	++_current_chunk;
	return &chunk;

	throw std::logic_error("Unimplemented");
}

static auto pki_config()
{
	QString pkidir("/home/fchapelle/dev/qtopcua/lay2form/pkidir");
	QOpcUaPkiConfiguration pkiConfig;
	pkiConfig.setClientCertificateFile(pkidir + "/own/certs/lay2form_client_certificate.der");
	pkiConfig.setPrivateKeyFile(pkidir + "/own/private/lay2form_client_private_key.pem");
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

void PVRush::PVOpcUaSource::connect_to_server()
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

	UA_StatusCode retval = UA_Client_connect_username(
	    client, _query.infos().get_host().toUtf8().data(),
	    _query.infos().get_login().toUtf8().data(), _query.infos().get_password().toUtf8().data());

	qDebug() << "UA_Client_connect_username:" << QDBGSTS(retval);

	/* Read historical values (uint32) */
	printf("\nStart historical read:\n");
	// UA_NodeId node = UA_NODEID_STRING(2, "MyLevel");
	UA_NodeId node = UA_NODEID_NUMERIC(2, 2021);
	retval = UA_Client_HistoryRead_raw(client, &node, readHist, UA_DateTime_fromUnixTime(0),
	                                   UA_DateTime_now(), UA_STRING_NULL, false, 1000,
	                                   UA_TIMESTAMPSTORETURN_BOTH, (void*)UA_FALSE);

	// retval = UA_Client_HistoryRead_atTime(
	//     client, &node, readHist, UA_DateTime_fromUnixTime(132113687876623850),
	//     UA_STRING_NULL, false, UA_TIMESTAMPSTORETURN_BOTH, (void*)UA_FALSE);

	// if (retval != UA_STATUSCODE_GOOD) {
	// 	printf("Failed. %s\n", UA_StatusCode_name(retval));
	// }

	UA_Client_disconnect(client);
	UA_Client_delete(client);
	QDBGF_END
}
