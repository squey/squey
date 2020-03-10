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
			_data[i].type = data_type;
			qDebug() << "PVOpcUaSource: node " << i << " has datatype " << data_type->typeName;
		} else {
			_data[i].type = &UA_TYPES[UA_TYPES_STRING];
			qDebug() << "PVOpcUaSource: Unknown type:" << deserialized_query[3 * i + 1];
		}
	}

	setup_query();
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

void PVRush::PVOpcUaSource::download_interval()
{
	std::vector<bool> has_data(_query_nb_of_times, false);

	for (size_t node = 0; node < _nodes_count; ++node) {
		auto& node_data = _data[node];
		const size_t elm_size = node_data.type->memSize;
		node_data.values.reserve(elm_size * _query_nb_of_times);

		UA_DateTime current_time = _query_start;
		size_t current_time_index = 0;
		_api.read_node_history(
		    _node_ids[node], _query_start, _query_end, 10000,
		    [this, &has_data, elm_size, &node_data, &current_time,
		     &current_time_index](UA_HistoryData* data) {
			    for (size_t i = 0; i < data->dataValuesSize; ++i) {
				    auto& dataval = data->dataValues[i];
				    if (dataval.value.type != node_data.type) {
					    qDebug() << "Row has bad data type" << dataval.value.type->typeName
					             << "(expected" << node_data.type->typeName << ")";
					    continue;
				    }
				    if (dataval.sourceTimestamp < current_time) {
					    if (node_data.values.empty()) {
						    qDebug() << __func__ << __LINE__;
						    PVOpcUaAPI::print_datetime(dataval.sourceTimestamp);
						    PVOpcUaAPI::print_datetime(current_time);
						    qDebug() << "Should not happen, wrong data or wrong logic.";
						    continue;
					    }
					    // Keep the last data for the interval
					    memcpy(node_data.values.data() + node_data.values.size() - elm_size, dataval.value.data,
					           elm_size);
					    continue;
				    }
				    // Fill the voids with zero or copy the last known element
				    while (dataval.sourceTimestamp >= current_time + _query_interval and
				           current_time < _query_end) {
					    const size_t old_size = node_data.values.size();
					    node_data.values.resize(old_size + elm_size);
					    if (node_data.values.empty()) {
						    memset(node_data.values.data() + old_size, 0, elm_size);
					    } else {
						    memcpy(node_data.values.data() + old_size,
						           node_data.values.data() + old_size - elm_size, elm_size);
					    }
					    current_time += _query_interval;
					    ++current_time_index;
				    }
				    // Copy the data for the element
				    const size_t old_size = node_data.values.size();
				    node_data.values.resize(old_size + elm_size);
				    memcpy(node_data.values.data() + old_size, dataval.value.data, elm_size);
				    has_data[current_time_index] = true;
				    current_time += _query_interval;
				    ++current_time_index;
			    }
			    return true;
		    });
		// Zero the rest or copy the last known element
		if (node_data.values.empty()) {
			node_data.values.resize(elm_size * _query_nb_of_times, 0);
		} else {
			const size_t old_size = node_data.values.size();
			node_data.values.resize(elm_size * _query_nb_of_times);
			for (size_t i = old_size; i < elm_size * _query_nb_of_times; i += elm_size) {
				memcpy(node_data.values.data() + i, node_data.values.data() + i - elm_size, elm_size);
			}
		}
	}

	for (size_t node = 0; node < _nodes_count; ++node) {
		auto& node_data = _data[node];
		const size_t elm_size = node_data.type->memSize;
		size_t consolidated_data = 0;
		for (size_t i = 0; i < has_data.size();) {
			const size_t yes_data_start = i;
			while (i < has_data.size() and has_data[i] == true) {
				++i;
			}
			memmove(node_data.values.data() + consolidated_data * elm_size,
			        node_data.values.data() + yes_data_start * elm_size, (i - yes_data_start) * elm_size);
			consolidated_data += i - yes_data_start;
			qDebug() << "Consolidated:" << consolidated_data << "/" << yes_data_start << "/" << i;
			while (i < has_data.size() and has_data[i] == false) {
				++i;
			}
		}
		node_data.values.resize(consolidated_data * elm_size);
		assert(node_data.values.size() == std::count(begin(has_data), end(has_data), true) * elm_size);
	}

	fill_sourcetime_interval(_query_start, has_data);
}

void PVRush::PVOpcUaSource::download_full()
{
	for (size_t node = 0; node < _nodes_count; ++node) {
		auto& node_data = _data[node];
		const size_t elm_size = node_data.type->memSize;

		_api.read_node_history(
		    _node_ids[node], _query_start, _query_end, 10000,
		    [this, elm_size, &node_data](UA_HistoryData* data) {
			    for (size_t i = 0; i < data->dataValuesSize; ++i) {
				    auto& dataval = data->dataValues[i];
				    if (dataval.value.type != node_data.type) {
					    qDebug() << "Row has bad data type" << dataval.value.type->typeName
					             << "(expected" << node_data.type->typeName << ")";
					    continue;
				    }
				    // Copy the data for the element
				    const size_t old_size = node_data.values.size();
				    node_data.values.resize(old_size + elm_size);
				    memcpy(node_data.values.data() + old_size, dataval.value.data, elm_size);
				    node_data.datetimes.push_back(dataval.sourceTimestamp);
			    }
			    return true;
		    });
	}

	std::vector<size_t> counters(_nodes_count, 0);
	std::vector<UA_DateTime> consolidated_datetimes;

	while (true /*at least one counter is < node_data.datetimes.size()*/) {
		auto min_node_it =
		    std::min_element(begin(_data), end(_data), [this, &counters](auto& a, auto& b) {
			    auto& a_counter = counters[std::distance(_data.data(), &a)];
			    auto& b_counter = counters[std::distance(_data.data(), &b)];
			    if (b_counter >= b.datetimes.size()) {
				    return true;
			    }
			    if (a_counter >= a.datetimes.size()) {
				    return false;
			    }
			    return a.datetimes[a_counter] < b.datetimes[b_counter];
		    });
		auto& min_node_counter = counters[std::distance(begin(_data), min_node_it)];
		if (min_node_counter >= min_node_it->datetimes.size()) {
			break;
		}
		auto min_datetime = min_node_it->datetimes[min_node_counter];
		consolidated_datetimes.push_back(min_datetime);
		for (size_t node = 0; node < _nodes_count; ++node) {
			if (_data[node].datetimes[counters[node]] <= min_datetime) {
				++counters[node];
			}
		}
	}

	for (size_t node = 0; node < _nodes_count; ++node) {
		auto& node_data = _data[node];
		const size_t elm_size = node_data.type->memSize;
		std::vector<uint8_t> consolidated_datavalues;
		consolidated_datavalues.reserve(consolidated_datetimes.size() * elm_size);
		size_t index_data = 0;
		for (auto& datetime : consolidated_datetimes) {
			const size_t old_size = consolidated_datavalues.size();
			consolidated_datavalues.resize(old_size + elm_size);
			if (node_data.datetimes[index_data] == datetime) {
				memcpy(consolidated_datavalues.data() + old_size,
				       node_data.values.data() + index_data * elm_size, elm_size);
				++index_data;
			} else { // node_data.datetimes[index_data] > datetime
				if (index_data == 0) {
					memset(consolidated_datavalues.data() + old_size, 0, elm_size);
				} else {
					memcpy(consolidated_datavalues.data() + old_size,
					       node_data.values.data() + (index_data - 1) * elm_size, elm_size);
				}
			}
		}
		std::swap(consolidated_datavalues, node_data.values);
	}

	fill_sourcetime_full(consolidated_datetimes);
}

PVCore::PVBinaryChunk* PVRush::PVOpcUaSource::operator()()
{
	if (_current_chunk == 0) {
		//download_interval();
		download_full();
	}

	if (_current_chunk > 0) {
		return nullptr;
	}

	auto chsize = _data[_current_chunk].values.size() / _data[_current_chunk].type->memSize;

	PVCore::PVBinaryChunk& chunk = *_chunks.emplace_back(
	    std::make_unique<PVCore::PVBinaryChunk>(_nodes_count + 1, chsize, _sourcetimes_current));

	chunk.set_raw_column_chunk(PVCol(0), _sourcetimes.data() + _sourcetimes_current, chsize,
	                           sizeof(boost::posix_time::ptime), "datetime_us");

	for (size_t i = 0; i < _nodes_count; ++i) {
		auto pvcop_type = PVRush::PVOpcUaAPI::pvcop_type(_data[i].type->typeIndex);
		qDebug() << "PVCOPTYPE:" << pvcop_type;
		chunk.set_raw_column_chunk(PVCol(1 + i), _data[i].values.data(), chsize,
		                           _data[i].type->memSize, pvcop_type);
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

void PVRush::PVOpcUaSource::setup_query()
{
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
	_query_interval = 200*UA_DATETIME_MSEC;
	_query_nb_of_times = (_query_end - _query_start) / _query_interval + 1;
	qDebug() << "QUERY_START";
	PVOpcUaAPI::print_datetime(_query_start);
	qDebug() << "QUERY_END";
	PVOpcUaAPI::print_datetime(_query_end);
	qDebug() << _query_start << _query_end << _query_nb_of_times;
}

void PVRush::PVOpcUaSource::fill_sourcetime_interval(UA_DateTime start_time,
                                                     std::vector<bool> const& has_data)
{
	static const UA_DateTimeStruct time_zero_dts = UA_DateTime_toStruct(0);
	static const boost::posix_time::ptime time_zero(
	    boost::gregorian::date(time_zero_dts.year, time_zero_dts.month, time_zero_dts.day));

	_sourcetimes.reserve(std::count(begin(has_data), end(has_data), true));

	for (size_t i = 0; i < has_data.size(); ++i) {
		if (has_data[i] == true) {
			_sourcetimes.emplace_back(
				time_zero +
				boost::posix_time::microsec((start_time + (i * _query_interval)) / UA_DATETIME_USEC));
		}
	}
}

void PVRush::PVOpcUaSource::fill_sourcetime_full(std::vector<UA_DateTime> const& datetimes)
{
	static const UA_DateTimeStruct time_zero_dts = UA_DateTime_toStruct(0);
	static const boost::posix_time::ptime time_zero(
	    boost::gregorian::date(time_zero_dts.year, time_zero_dts.month, time_zero_dts.day));

	_sourcetimes.reserve(datetimes.size());

	for (auto& datetime : datetimes) {
		_sourcetimes.emplace_back(
		    time_zero +
		    boost::posix_time::microsec(datetime / UA_DATETIME_USEC));
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
