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
	bool end_of_data = true;

	std::vector<bool> has_data(_query_nb_of_times, false);

	for (size_t node = 0; node < _nodes_count; ++node) {
		auto& node_data = _data[node];
		const size_t elm_size = node_data.type->memSize;
		node_data.values.reserve(elm_size * _query_nb_of_times);

		if (not node_data.has_more) {
			continue;
		}

		UA_DateTime current_time = _query_start + _current_chunk * _chunk_size;
		size_t current_time_index = 0;
		_api.read_node_history(
		    _node_ids[node], current_time, _query_end, _chunk_size,
		    [this, &has_data, elm_size, &node_data, &current_time,
		     &current_time_index](UA_HistoryData* data, bool has_more) {
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
				node_data.has_more = has_more and data->dataValuesSize < _chunk_size;
			    return false;
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

		if (node_data.has_more) {
			end_of_data = false;
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
			// qDebug() << "Consolidated:" << consolidated_data << "/" << yes_data_start << "/" << i;
			while (i < has_data.size() and has_data[i] == false) {
				++i;
			}
		}
		node_data.values.resize(consolidated_data * elm_size);
		assert(node_data.values.size() == std::count(begin(has_data), end(has_data), true) * elm_size);
		node_data.chunk_values = std::move(node_data.values);
	}

	fill_sourcetime_interval(_query_start, has_data);

	if (end_of_data) {
		_chunk_size = 0;
	}
}

void PVRush::PVOpcUaSource::download_full()
{
	bool end_of_data = true;

	for (size_t node = 0; node < _nodes_count; ++node) {
		auto& node_data = _data[node];
		const size_t elm_size = node_data.type->memSize;

		if (not node_data.has_more) {
			continue;
		}

		UA_DateTime chunk_start = node_data.read_index == 0 ? _query_start
								: node_data.datetimes.back() + UA_DATETIME_USEC;

		_api.read_node_history(
		    _node_ids[node], chunk_start, _query_end, _chunk_size,
		    [this, elm_size, &node_data](UA_HistoryData* data, bool has_more) {
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
				node_data.has_more = has_more;
			    return false;
		    });
		
		if (node_data.has_more) {
			end_of_data = false;
		}
	}

	auto consolidated_datetimes = consolidate_datetimes_full();

	consolidate_values_full(consolidated_datetimes);
	fill_sourcetime_full(consolidated_datetimes);

	if (end_of_data and consolidated_datetimes.size() < _chunk_size) {
		_chunk_size = 0;
	}
}

auto PVRush::PVOpcUaSource::operator()() -> PVOpcUaBinaryChunk*
{
	if (_chunk_size == 0) {
		return nullptr;
	}

	//download_interval();
	download_full();

	size_t chsize = _sourcetimes.size();

	auto chunk = std::make_unique<PVOpcUaBinaryChunk>(_nodes_count, chsize, 
		                       _sourcetimes_current, std::move(_sourcetimes));

	for (size_t i = 0; i < _nodes_count; ++i) {
		chunk->set_node_values(i, std::move(_data[i].chunk_values), _data[i].type);
	}

	pvlogger::debug() << "Generated chunk of " << chsize
	                  << " lines starting line " << _sourcetimes_current
					  << std::endl;

	_sourcetimes_current += chsize;
	++_current_chunk;
	return chunk.release();
}

void PVRush::PVOpcUaSource::setup_query()
{
	UA_DateTime first_historical_datetime = UA_DateTime_now();
	for (auto& node_id : _node_ids) {
		try {
			auto node_first_datetime = _api.first_historical_datetime(node_id);
			// qDebug() << node_id << " first datetime:";
			// PVOpcUaAPI::print_datetime(node_first_datetime);
			if (node_first_datetime < first_historical_datetime) {
				first_historical_datetime = node_first_datetime;
			}
		} catch (std::system_error& error) {
			pvlogger::error() << "Fetching first historical datetime for node " << node_id.toStdString()
			                  << " : " << error.code() << " " << error.what() << std::endl;
		}
	}
	_query_start = first_historical_datetime;
	//_query_start = UA_DateTime(UA_DATETIME_SEC);
	//_query_start = UA_DateTime_now() - 6000 * UA_DATETIME_SEC;
	_query_end = UA_DateTime_now();
	_query_interval = 200*UA_DATETIME_MSEC;
	_query_nb_of_times = (_query_end - _query_start) / _query_interval + 1;
}

void PVRush::PVOpcUaSource::fill_sourcetime_interval(UA_DateTime start_time,
                                                     std::vector<bool> const& has_data)
{
	_sourcetimes = decltype(_sourcetimes){};
	_sourcetimes.reserve(std::count(begin(has_data), end(has_data), true));

	for (size_t i = 0; i < has_data.size(); ++i) {
		if (has_data[i] == true) {
			_sourcetimes.emplace_back(PVOpcUaAPI::to_ptime(start_time + (i * _query_interval)));
		}
	}
}

void PVRush::PVOpcUaSource::fill_sourcetime_full(std::vector<UA_DateTime> const& datetimes)
{
	_sourcetimes = decltype(_sourcetimes){};
	_sourcetimes.reserve(datetimes.size());

	for (auto& datetime : datetimes) {
		_sourcetimes.emplace_back(PVOpcUaAPI::to_ptime(datetime));
	}
}

auto PVRush::PVOpcUaSource::consolidate_datetimes_full() const -> std::vector<UA_DateTime>
{
	std::vector<UA_DateTime> consolidated_datetimes;

	std::vector<size_t> counters(_nodes_count, 0);
	for (size_t node = 0; node < _nodes_count; ++node) {
		counters[node] = _data[node].read_index;
	}

	while (consolidated_datetimes.size() < _chunk_size * _nodes_count) {
		UA_DateTime min_node_datetime = _query_end;
		// Find minimum datetime
		for (size_t node = 0; node < _nodes_count; ++node) {
			if (counters[node] < _data[node].datetimes.size()) {
				auto node_datetime = _data[node].datetimes[counters[node]];
				if (node_datetime < min_node_datetime) {
					min_node_datetime = node_datetime;
				}
			}
		}
		// Break when all counters have reached their limit
		if (min_node_datetime == _query_end) {
			break;
		}
		// Push datetime
		consolidated_datetimes.push_back(min_node_datetime);
		// Advance counters of minimum datetime nodes
		for (size_t node = 0; node < _nodes_count; ++node) {
			if (counters[node] < _data[node].datetimes.size()) {
				auto node_datetime = _data[node].datetimes[counters[node]];
				assert(node_datetime >= min_node_datetime);
				if (node_datetime == min_node_datetime) {
					++counters[node];
				}
			}
		}
	}

	return consolidated_datetimes;
}

void PVRush::PVOpcUaSource::consolidate_values_full(std::vector<UA_DateTime> const& consolidated_datetimes)
{
	for (size_t node = 0; node < _nodes_count; ++node) {
		auto& node_data = _data[node];
		const size_t elm_size = node_data.type->memSize;
		auto& consolidated_datavalues = node_data.chunk_values;
		consolidated_datavalues.clear();
		consolidated_datavalues.reserve(consolidated_datetimes.size() * elm_size);
		size_t& read_index = node_data.read_index;
		for (auto& datetime : consolidated_datetimes) {
			const size_t old_size = consolidated_datavalues.size();
			consolidated_datavalues.resize(old_size + elm_size);
			if (node_data.datetimes[read_index] == datetime) {
				memcpy(consolidated_datavalues.data() + old_size,
				       node_data.values.data() + read_index * elm_size, elm_size);
				++read_index;
			} else { // node_data.datetimes[read_index] > datetime
				if (read_index == 0) {
					memset(consolidated_datavalues.data() + old_size, 0, elm_size);
				} else {
					memcpy(consolidated_datavalues.data() + old_size,
					       node_data.values.data() + (read_index - 1) * elm_size, elm_size);
				}
			}
		}
		if (read_index > 0) {
			auto erase_front_bytes = [](auto& container, size_t last_index){
				container.erase(begin(container), std::next(begin(container), last_index));
			};
			erase_front_bytes(node_data.datetimes, read_index - 1);
			erase_front_bytes(node_data.values, (read_index - 1) * elm_size);
			read_index = 1;
		}
	}
}
