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
    : _query(dynamic_cast<PVOpcUaQuery&>(*input)), _api(_query.infos())
{
	qDebug() << "PVOpcUaSource::operator()";
	qDebug() << _query.human_name() << _query.get_query() << _query.get_query_type();

	qDebug() << "Infos:" << _query.infos().get_host();

	auto serialized_query = _query.get_query();
	auto deserialized_query = serialized_query.split(QRegularExpression("\\;\\$\\;"));
	qDebug() << deserialized_query;
	_nodes_count = deserialized_query.size() / 3;
	_data.resize(_nodes_count);
	for (size_t i = 0; i < _nodes_count; ++i) {
		// configure node per node
		_node_ids.push_back(deserialized_query[3 * i]);
		auto node_id_open62541 = PVOpcUaAPI::NodeId(deserialized_query[3 * i + 1]).open62541();
		if (auto* data_type = UA_findDataType(&node_id_open62541)) {
			_data[i].second = data_type;
			qDebug() << "PVOpcUaSource: node " << i << " has datatype " << data_type->typeName;
		} else {
			_data[i].second = &UA_TYPES[UA_TYPES_STRING];
			qDebug() << "PVOpcUaSource: Unknown type:" << deserialized_query[3 * i + 1];
		}
	}
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
	if (_current_chunk == 0) {
		fill_sourcetime();

		for (size_t node = 0; node < _nodes_count; ++node) {
			auto& [node_data, node_datatype] = _data[node];
			const size_t elm_size = node_datatype->memSize;
			node_data.reserve(elm_size * _query_nb_of_times);

			UA_DateTime current_time = _query_start;
			size_t current_time_index = 0;
			_api.read_node_history(
			    _node_ids[node], _query_start, _query_end,
			    [this, elm_size, &node_data, &node_datatype, &current_time,
			     &current_time_index](UA_HistoryData* data) {
				    for (size_t i = 0; i < data->dataValuesSize; ++i) {
					    auto& dataval = data->dataValues[i];
					    if (dataval.value.type != node_datatype) {
						    qDebug() << "Row has bad data type" << dataval.value.type->typeName
						             << "(expected" << node_datatype->typeName << ")";
						    continue;
					    }
						if (dataval.sourceTimestamp < current_time) {
							if (node_data.empty()) {
								qDebug() << __func__ << __LINE__;
								PVOpcUaAPI::print_datetime(dataval.sourceTimestamp);
								PVOpcUaAPI::print_datetime(current_time);
							    qDebug() << "Should not happen, wrong data or wrong logic.";
							    continue;
						    }
						    // Keep the last data for the interval
						    memcpy(node_data.data() + node_data.size() - elm_size,
						           dataval.value.data, elm_size);
						    continue;
						}
					    // Fill the voids with zero or copy the last known element
					    while (dataval.sourceTimestamp >= current_time + _query_interval and
								current_time < _query_end) {
							const size_t old_size = node_data.size();
							node_data.resize(old_size + elm_size);
							if (node_data.empty()) {
								memset(node_data.data() + old_size, 0, elm_size);
							} else {
								memcpy(node_data.data() + old_size,
										node_data.data() + old_size - elm_size,
										elm_size);
							}
							current_time += _query_interval;
							++current_time_index;
						}
						// Copy the data for the element
						const size_t old_size = node_data.size();
						node_data.resize(old_size + elm_size);
						memcpy(node_data.data() + old_size, dataval.value.data,
								elm_size);
						current_time += _query_interval;
						++current_time_index;
				    }
				    return true;
			    });
			// Zero the rest or copy the last known element
			if (node_data.empty()) {
				node_data.resize(elm_size * _query_nb_of_times, 0);
			} else {
				const size_t old_size = node_data.size();
				node_data.resize(elm_size * _query_nb_of_times);
				for (size_t i = old_size; i < elm_size * _query_nb_of_times; i += elm_size) {
					memcpy(node_data.data() + i,
					       node_data.data() + i - elm_size, elm_size);
				}
			}
		}
	}

	if (_current_chunk > 0) {
		return nullptr;
	}

	auto chsize = _data[_current_chunk].first.size() / _data[_current_chunk].second->memSize;

	PVCore::PVBinaryChunk& chunk = *_chunks.emplace_back(
	    std::make_unique<PVCore::PVBinaryChunk>(_nodes_count + 1, chsize, _sourcetimes_current));

	chunk.set_raw_column_chunk(PVCol(0), _sourcetimes.data() + _sourcetimes_current, chsize,
	                           sizeof(boost::posix_time::ptime), "datetime_us");

	for (size_t i = 0; i < _nodes_count; ++i) {
		auto pvcop_type = PVRush::PVOpcUaAPI::pvcop_type(_data[i].second->typeIndex);
		qDebug() << "PVCOPTYPE:" << pvcop_type;
		chunk.set_raw_column_chunk(PVCol(1 + i), _data[i].first.data(), chsize,
		                           _data[i].second->memSize, pvcop_type);
		// if (i == _current_chunk) {
		// 	chunk.set_raw_column_chunk(PVCol(1 + i),
		// 	                           _data[i].first.data(), chsize,
		// 	                           _data[i].second->memSize, pvcop_type);
		// } else {
		// 	chunk.set_invalid_column(PVCol(1 + i));
		// 	chunk.set_raw_column_chunk(PVCol(1 + i), _empty_column.data(), chsize,
		// 	                           _data[i].second->memSize, pvcop_type);
		// }
	}
	chunk.set_rows_count(chsize);
	qDebug() << "chunk filled";
	//_sourcetimes_current += chsize;
	++_current_chunk;
	return &chunk;

	throw std::logic_error("Unimplemented");
}

void PVRush::PVOpcUaSource::fill_sourcetime()
{
	static const UA_DateTimeStruct time_zero_dts = UA_DateTime_toStruct(0);
	static const boost::posix_time::ptime time_zero(
	    boost::gregorian::date(time_zero_dts.year, time_zero_dts.month, time_zero_dts.day));

	UA_DateTime first_historical_datetime = UA_DateTime_now();
	std::cout << "node_ids:" << _node_ids.size() << " now():" << first_historical_datetime
	          << std::endl;
	for (auto& node_id : _node_ids) {
		auto node_first_datetime = _api.first_historical_datetime(node_id);
		qDebug() << node_id << " first datetime:";
		PVOpcUaAPI::print_datetime(node_first_datetime);
		if (node_first_datetime < first_historical_datetime) {
			first_historical_datetime = node_first_datetime;
		}
	}
	_query_start = first_historical_datetime;

	//_query_start = UA_DateTime(0);// UA_DateTime_now() - 6000 * UA_DATETIME_SEC;
	_query_end = UA_DateTime_now();
	_query_interval = 20 * UA_DATETIME_SEC;
	_query_nb_of_times = (_query_end - _query_start) / _query_interval + 1;
	qDebug() << "QUERY_START";
	PVOpcUaAPI::print_datetime(_query_start);
	qDebug() << "QUERY_END";
	PVOpcUaAPI::print_datetime(_query_end);
	qDebug() << _query_start << _query_end << _query_nb_of_times;

	_sourcetimes.reserve(_query_nb_of_times);

	for (size_t i = 0; i < _query_nb_of_times; ++i) {
		_sourcetimes.emplace_back(
		    time_zero +
		    boost::posix_time::microsec((_query_start + (i * _query_interval)) / UA_DATETIME_USEC));
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
