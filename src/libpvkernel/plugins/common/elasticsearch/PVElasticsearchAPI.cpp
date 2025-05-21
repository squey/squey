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

#include "PVElasticsearchAPI.h"

#include <curl/curl.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVFormat_types.h>
#include <pvcop/db/array.h>
#include <pvcop/types/factory.h>
#include <pvcop/formatter_desc.h>
#include <rapidjson/pointer.h>
#include <qstring.h>
#include <rapidjson/allocators.h>
#include <rapidjson/encodings.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/reader.h>
#include <rapidjson/stream.h>
#include <stdint.h>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/detail/classification.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <sstream>
#include <cerrno>
#include <unordered_set>
#include <numeric>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <tuple>

#include "PVElasticsearchInfos.h"
#include "PVElasticsearchQuery.h"
#include "PVElasticsearchSAXParser.h"
#include "pvcop/types/formatter/formatter_interface.h"
#include "pvkernel/core/PVVersion.h"
#include "pvlogger.h"

static constexpr const size_t DEFAULT_SCROLL_SIZE = 10000;
static constexpr const char SCROLL_TIMEOUT[] = "1m";

/**
 * cURL callback function used to fill the buffer returned by perform_query
 */
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp)
{
	((std::string*)userp)->append((char*)contents, size * nmemb);

	return size * nmemb;
}

/**
 * Helper function to retrieve errors raised by Elasticsearch
 */
bool PVRush::PVElasticsearchAPI::has_error(const rapidjson::Document& json,
                                           std::string* error) const
{
	if (json.IsObject() and json.HasMember("error")) {
		if (error) {
			if (_version < PVCore::PVVersion(2, 0, 0)) {
				*error = json["error"].GetString();
			} else {
				const auto& reason = json["error"]["root_cause"][0]["reason"];
				*error = reason.IsNull() ? "Unspecified error" : reason.GetString();
			}
		}
		return true;
	}

	return false;
}

PVRush::PVElasticsearchAPI::PVElasticsearchAPI(const PVRush::PVElasticsearchInfos& infos)
    : _curl(nullptr), _infos(infos)
{
	_curl = curl_easy_init();
	std::string content_type_header{"Content-Type: application/json"};
	_curl_headers = curl_slist_append(_curl_headers, content_type_header.c_str());
	_version = version();
}

PVRush::PVElasticsearchAPI::~PVElasticsearchAPI()
{
	curl_slist_free_all(_curl_headers);
	curl_easy_cleanup(_curl);
}

bool PVRush::PVElasticsearchAPI::check_connection(std::string* error /* =  nullptr */) const
{
	std::string json_buffer;

	prepare_query(_curl, socket());
	if (perform_query(_curl, json_buffer, error)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		return not has_error(json, error);
	}

	return false;
}

PVCore::PVVersion PVRush::PVElasticsearchAPI::version() const
{
	std::string json_buffer;
	std::string error;

	prepare_query(_curl, socket());
	perform_query(_curl, json_buffer, &error);

	if (error.empty()) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());
		if (json.IsObject() and json.HasMember("version") and json["version"].HasMember("number")) {
			return {json["version"]["number"].GetString()};
		}
	}

	// if we don't know the version, assume it's the latest
	return {(size_t)-1, (size_t)-1, (size_t)-1};
}

size_t PVRush::PVElasticsearchAPI::shards_count(const std::string& index,
                                                std::string* error /*= nullptr*/) const
{
	std::string json_buffer;
	std::string url = socket() + "/" + index + "/_stats?filter_path=_shards.successful";

	prepare_query(_curl, url);
	if (perform_query(_curl, json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (not has_error(json, error) and json.HasMember("_shards") and
		    json["_shards"].HasMember("successful")) {

			return json["_shards"]["successful"].GetUint();
		}
	}

	return 0;
}

size_t PVRush::PVElasticsearchAPI::max_result_window(const std::string& index) const
{
	size_t max_result_window = DEFAULT_SCROLL_SIZE;

	std::string json_buffer;

	std::string url = socket() + "/" + index + "/_settings";
	prepare_query(_curl, url);
	if (perform_query(_curl, json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (not has_error(json) and json.HasMember(index.c_str()) and
		    json[index.c_str()].HasMember("settings") and
		    json[index.c_str()]["settings"].HasMember("index") and
		    json[index.c_str()]["settings"]["index"].HasMember("max_result_window")) {
			const std::string max_result_window_str =
			    json[index.c_str()]["settings"]["index"]["max_result_window"].GetString();

			errno = 0;
			char* tmp;
			size_t value = std::strtoull(max_result_window_str.c_str(), &tmp, 10);
			if (not(tmp == max_result_window_str.c_str() || *tmp != '\0' || errno != 0)) {
				max_result_window = value;
			}
		}
	}

	return max_result_window;
}

static std::string get_filter_path_from_base(CURL* curl,
                                             const std::string& filter_path,
                                             const std::string& base = {},
                                             const std::string& separator = ".")
{
	std::vector<std::string> relative_columns;
	boost::algorithm::split(relative_columns, filter_path, boost::is_any_of(","));

	std::vector<std::string> absolute_columns;
	for (std::string& column : relative_columns) {
		// URL encode columns
		char* url_encoded_column = curl_easy_escape(curl, column.c_str(), column.size());
		column = url_encoded_column;
		curl_free(url_encoded_column);

		PVCore::replace(column, ".", separator);
		absolute_columns.emplace_back(base.empty() ? column : base + "." + column);
	}

	return boost::algorithm::join(absolute_columns, ",");
}

static std::vector<std::string> get_pointers_type_from_base(const std::string& filter_path,
                                                            const std::string& base = {})
{
	std::vector<std::string> relative_columns;
	boost::algorithm::split(relative_columns, filter_path, boost::is_any_of(","));

	std::vector<std::string> absolute_columns;
	for (std::string& column : relative_columns) {
		PVCore::replace(column, ".", "/properties/");
		absolute_columns.emplace_back(base.empty() ? column : base + "/" + column + "/type");
	}

	return absolute_columns;
}

PVRush::PVElasticsearchAPI::indexes_t
PVRush::PVElasticsearchAPI::indexes(std::string* error /*= nullptr*/) const
{
	indexes_t indexes;
	std::string json_buffer;
	std::string url = socket() + "/_stats";

	prepare_query(_curl, url);
	if (perform_query(_curl, json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (has_error(json, error)) {
			return {};
		}

		rapidjson::Value& json_indexes = json["indices"];

		for (rapidjson::Value::ConstMemberIterator itr = json_indexes.MemberBegin();
		     itr != json_indexes.MemberEnd(); ++itr) {
			indexes.emplace_back(itr->name.GetString());
		}
	}

	std::sort(indexes.begin(), indexes.end());

	return indexes;
}

PVRush::PVElasticsearchAPI::aliases_t
PVRush::PVElasticsearchAPI::aliases(std::string* error /*= nullptr*/) const
{
	std::unordered_set<std::string> aliases_set;
	aliases_t aliases;
	std::string json_buffer;
	std::string url = socket() + "/_cat/aliases?format=json&filter_path=alias";

	prepare_query(_curl, url);
	if (perform_query(_curl, json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (json.IsNull() or has_error(json, error)) {
			return {};
		}

		for (rapidjson::SizeType i = 0; i < json.Size(); i++) {
			aliases_set.emplace(json[i]["alias"].GetString());
		}
	}

	aliases.assign(aliases_set.begin(), aliases_set.end());
	std::sort(aliases.begin(), aliases.end());

	return aliases;
}

/*
 * Recursively visit all the columns of the mapping
 *
 * @param json json object representing the mapping (possibily filtered by filter_path)
 * @param f the function that is called against each column
 */
static void visit_columns_rec(const rapidjson::Value& json,
                              const PVRush::PVElasticsearchAPI::visit_columns_f& f,
                              const std::string& parent_name = {})
{
	size_t children_count = json.MemberCount();
	size_t child_index = 0;
	for (auto p = json.MemberBegin(); p != json.MemberEnd(); ++p, child_index++) {
		const std::string& rel_name = p->name.GetString();
		const std::string& abs_name = parent_name.empty() ? rel_name : parent_name + "." + rel_name;

		const auto& field = json[rel_name.c_str()];
		if (field.HasMember("type")) {
			const std::string& type = field["type"].GetString();
			f(rel_name, abs_name, type, true, child_index == children_count - 1);
		} else if (field.HasMember("properties")) {
			const rapidjson::Value& properties = field["properties"];
			f(rel_name, abs_name, "", false, child_index == children_count - 1);
			visit_columns_rec(properties, f, abs_name);
		}
	}
}

void PVRush::PVElasticsearchAPI::visit_columns(const visit_columns_f& f,
                                               const std::string& filter_path /* = "" */,
                                               std::string* error /*= nullptr*/) const
{
	if (_infos.get_index().isEmpty()) {
		if (error) {
			*error = "No index specified";
		}
	}

	std::string json_buffer;
	std::string url = socket() + "/" + _infos.get_index().toStdString() + "/_mapping";

	prepare_query(_curl, url);
	if (perform_query(_curl, json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());
		if (has_error(json, error)) {
			return;
		}

		_mapping_type.clear();
		// for retro-compatibility purposes :
		// https://www.elastic.co/guide/en/elasticsearch/reference/current/removal-of-types.html
		{
			const rapidjson::Value& mappings =
			    json[_infos.get_index().toStdString().c_str()]["mappings"];
			if (mappings.MemberCount() == 1) {
				if (not mappings.HasMember("properties")) {
					_mapping_type = mappings.GetObj().MemberBegin()->name.GetString();
				}
			} else {
				if (_version < PVCore::PVVersion(7, 0, 0)) {
					if (not mappings.HasMember("_default_")) {
						_mapping_type = "_default_";
					}
				}
				else {
					_mapping_type = "";
				}
			}
		}

		// filter fields client-side because "filter_path" usage can cause
		// "too_long_frame_exception, An HTTP line is larger than 4096 bytes" error
		if (not filter_path.empty()) {
			const std::string& base =
			    std::string("/") + _infos.get_index().toStdString() + "/mappings/" +
			    (_mapping_type.empty() ? "" : (_mapping_type + "/")) + "properties";
			const std::vector<std::string>& pointers =
			    get_pointers_type_from_base(filter_path, base);
			rapidjson::Document filtered_json;
			for (const std::string& pointer : pointers) {
				const rapidjson::Value* filtered = rapidjson::Pointer(pointer.c_str()).Get(json);
				rapidjson::Pointer(pointer.c_str()).Set(filtered_json, *filtered);
			}
			json = std::move(filtered_json);
		}
		const rapidjson::Value& mappings =
		    json[_infos.get_index().toStdString().c_str()]["mappings"];

		visit_columns_rec(_mapping_type.empty() ? mappings["properties"]
		                                        : mappings[_mapping_type.c_str()]["properties"],
		                  f);
	}
}

PVRush::PVElasticsearchAPI::columns_t PVRush::PVElasticsearchAPI::format_columns(
    const std::string& filter_path /* = {} */, std::string* error /*= nullptr*/
    ) const
{
	auto map_type = [&](const std::string& type) -> std::string {
		const auto& it = types_mapping().find(type);
		if (it != types_mapping().end()) {
			return it->second;
		} else {
			// fallback type for unkown types
			return "string";
		}
	};

	columns_t cols;

	visit_columns(
	    [&](const std::string& /*rel_name*/, const std::string& abs_name, const std::string& type,
	        bool is_leaf, bool /*is_last_child*/) {
		    if (is_leaf) {
			    cols.emplace_back(abs_name, std::make_pair(map_type(type), ""));
		    }
	    },
	    filter_path, error);

	// Narrow numeric types based on aggregation max value
	narrow_numeric_types(cols);

	detect_time_formats(cols);

	return cols;
}

void PVRush::PVElasticsearchAPI::narrow_numeric_types(columns_t& cols) const
{
	std::unordered_map<std::string, std::string> numeric_types{{"long", "number_int64"},
	                                                           {"integer", "number_int32"},
	                                                           {"short", "number_int16"},
	                                                           {"byte", "number_int8"}};

	std::vector<std::tuple<std::string, int64_t, uint64_t>> types_ranges{
	    {{"number_uint8", std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()},
	     {"number_int8", std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()},
	     {"number_uint16", std::numeric_limits<uint16_t>::min(),
	      std::numeric_limits<uint16_t>::max()},
	     {"number_int16", std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()},
	     {"number_uint32", std::numeric_limits<uint32_t>::min(),
	      std::numeric_limits<uint32_t>::max()},
	     {"number_int32", std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()},
	     {"number_uint64", std::numeric_limits<uint64_t>::min(),
	      std::numeric_limits<uint64_t>::max()},
	     {"number_int64", std::numeric_limits<int64_t>::min(),
	      std::numeric_limits<int64_t>::max()}}};

	std::vector<size_t> numeric_col_indexes;
	for (size_t i = 0; i < cols.size(); i++) {
		auto& col = cols[i];
		if (numeric_types.find(col.second.first) != numeric_types.end()) {
			numeric_col_indexes.emplace_back(i);
		}
	}

	rapidjson::Document json;
	json.SetObject();

	rapidjson::Value aggs;
	aggs.SetObject();

	for (size_t index : numeric_col_indexes) {
		const std::string& fname = cols[index].first;

		rapidjson::Value field_name_min_str;
		field_name_min_str.SetString(fname.c_str(), json.GetAllocator());

		rapidjson::Value field_name_max_str;
		field_name_max_str.SetString(fname.c_str(), json.GetAllocator());

		rapidjson::Value fieldname_min;
		fieldname_min.SetObject();
		fieldname_min.AddMember("field", field_name_min_str, json.GetAllocator());

		rapidjson::Value min;
		min.SetObject();
		min.AddMember("min", fieldname_min, json.GetAllocator());

		rapidjson::Value fieldname_max;
		fieldname_max.SetObject();
		fieldname_max.AddMember("field", field_name_max_str, json.GetAllocator());

		rapidjson::Value max;
		max.SetObject();
		max.AddMember("max", fieldname_max, json.GetAllocator());

		rapidjson::Value min_field;
		min_field.SetString((fname + ".min").c_str(), json.GetAllocator());

		rapidjson::Value max_field;
		max_field.SetString((fname + ".max").c_str(), json.GetAllocator());

		aggs.AddMember(min_field, min, json.GetAllocator());
		aggs.AddMember(max_field, max, json.GetAllocator());
	}

	json.AddMember("aggs", aggs, json.GetAllocator());

	rapidjson::StringBuffer strbuf;
	rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
	json.Accept(writer);

	std::unordered_map<std::string, std::pair<double, double>> values_ranges;
	std::string json_buffer;
	std::string url = socket() + "/" + _infos.get_index().toStdString() + "/_search?size=0";
	prepare_query(_curl, url, strbuf.GetString());
	if (perform_query(_curl, json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (not json.HasMember("aggregations")) {
			return;
		}

		const rapidjson::Value& aggregations = json["aggregations"];
		for (auto agg = aggregations.MemberBegin(); agg != aggregations.MemberEnd(); ++agg) {
			const std::string& value_name = agg->name.GetString();
			const rapidjson::Value& value_json = aggregations[value_name.c_str()]["value"];
			if (not value_json.IsNumber()) {
				continue;
			}
			double value = value_json.GetDouble();

			const std::string& field_name = value_name.substr(0, value_name.size() - 4);
			const std::string& operation_name = value_name.substr(value_name.size() - 3);

			if (operation_name == "min") {
				values_ranges[field_name].first = value;
			} else {
				values_ranges[field_name].second = value;
			}
		}
	}

	for (size_t index : numeric_col_indexes) {
		auto& fname = cols[index].first;
		const auto& it = values_ranges.find(fname);
		if (it != values_ranges.end()) {
			double value_min = it->second.first;
			double value_max = it->second.second;

			std::string smallest_type;
			for (const auto& type_range : types_ranges) {
				const std::string& type_name = std::get<0>(type_range);
				int64_t type_min = std::get<1>(type_range);
				uint64_t type_max = std::get<2>(type_range);

				if (value_min >= type_min and value_max <= type_max) {
					smallest_type = type_name;
					break;
				}
			}
			cols[index].second.first = smallest_type;
		} else { // fallback to default type
			cols[index].second.first = numeric_types.find(cols[index].second.first)->second;
		}
	}

	/*
	POST /<index_name>/_search?size=0
	{
	  "aggs" : {
	    "<field_name>.max" : { "max" : { "field" : "<field_name>" } },
	    "<field_name>.min" : { "min" : { "field" : "<field_name>" } }
	  }
	}
	*/
}

void PVRush::PVElasticsearchAPI::detect_time_formats(columns_t& cols) const
{
	// https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-date-format.html
	std::multimap<std::string, std::string> dateformat_map = {
	    {"epoch_millis", "epoch.S"},
	    {"epoch_second", "epoch"},
	    {"date_optional_time", "yyyy-MM-dd'T'HH:mm:ss.SSSZ"},
	    {"date_optional_time", "yyyy-MM-dd"},
	    {"strict_date_optional_time", "yyyy-MM-dd'T'HH:mm:ss.SSSZ"},
	    {"strict_date_optional_time", "yyyy-MM-dd"},
	    {"basic_date", "yyyyMMdd"},
	    {"basic_date_time", "yyyyMMdd'T'HHmmss.SSSZ"},
	    {"basic_date_time_no_millis", "yyyyMMdd'T'HHmmssZ"},
	    {"basic_ordinal_date", "yyyyDDD"},
	    {"basic_ordinal_date_time", "yyyyDDD'T'HHmmss.SSSZ"},
	    {"basic_ordinal_date_time_no_millis", "yyyyDDD'T'HHmmssZ"},
	    {"basic_time", "HHmmss.SSSZ"},
	    {"basic_time_no_millis", "HHmmssZ"},
	    {"basic_t_time", "'T'HHmmss.SSSZ"},
	    {"basic_t_time_no_millis", "T'HHmmssZ"},
	    {"basic_week_date", "xxxx'W'wwe"},
	    {"strict_basic_week_date", "xxxx'W'wwe"},
	    {"basic_week_date_time", "xxxx'W'wwe'T'HHmmss.SSSZ"},
	    {"strict_basic_week_date_time", "xxxx'W'wwe'T'HHmmss.SSSZ"},
	    {"basic_week_date_time_no_millis", "xxxx'W'wwe'T'HHmmssZ"},
	    {"strict_basic_week_date_time_no_millis", "xxxx'W'wwe'T'HHmmssZ"},
	    {"date", "strict_date"},
	    {"date_hour", "yyyy-MM-dd'T'HH"},
	    {"strict_date_hour", "yyyy-MM-dd'T'HH"},
	    {"date_hour_minute", "yyyy-MM-dd'T'HH:mm"},
	    {"strict_date_hour_minute", "yyyy-MM-dd'T'HH:mm"},
	    {"date_hour_minute_second", "yyyy-MM-dd'T'HH:mm:ss"},
	    {"strict_date_hour_minute_second", "yyyy-MM-dd'T'HH:mm:ss"},
	    {"date_hour_minute_second_fraction", "yyyy-MM-dd'T'HH:mm:ss.SSS"},
	    {"strict_date_hour_minute_second_fraction", "yyyy-MM-dd'T'HH:mm:ss.SSS"},
	    {"date_hour_minute_second_millis", "yyyy-MM-dd'T'HH:mm:ss.SSS"},
	    {"strict_date_hour_minute_second_millis", "yyyy-MM-dd'T'HH:mm:ss.SSS"},
	    {"date_time", "yyyy-MM-dd'T'HH:mm:ss.SSSZZ"},
	    {"strict_date_time", "yyyy-MM-dd'T'HH:mm:ss.SSSZZ"},
	    {"date_time_no_millis", "yyyy-MM-dd'T'HH:mm:ssZZ"},
	    {"strict_date_time_no_millis", "yyyy-MM-dd'T'HH:mm:ssZZ"},
	    {"hour", "HH"},
	    {"strict_hour", "HH"},
	    {"hour_minute", "HH:mm"},
	    {"strict_hour_minute", "HH:mm"},
	    {"hour_minute_second", "HH:mm:ss"},
	    {"strict_hour_minute_second", "HH:mm:ss"},
	    {"hour_minute_second_fraction", "HH:mm:ss.SSS"},
	    {"strict_hour_minute_second_fraction", "HH:mm:ss.SSS"},
	    {"hour_minute_second_millis", "HH:mm:ss.SSS"},
	    {"strict_hour_minute_second_millis", "HH:mm:ss.SSS"},
	    {"ordinal_date", "yyyy-DDD"},
	    {"strict_ordinal_date", "yyyy-DDD"},
	    {"ordinal_date_time", "yyyy-DDD'T'HH:mm:ss.SSSZZ"},
	    {"strict_ordinal_date_time", "yyyy-DDD'T'HH:mm:ss.SSSZZ"},
	    {"ordinal_date_time_no_millis", "yyyy-DDD'T'HH:mm:ssZZ"},
	    {"strict_ordinal_date_time_no_millis", "yyyy-DDD'T'HH:mm:ssZZ"},
	    {"time", "HH:mm:ss.SSSZZ"},
	    {"strict_time", "HH:mm:ss.SSSZZ"},
	    {"time_no_millis", "HH:mm:ssZZ"},
	    {"strict_time_no_millis", "HH:mm:ssZZ"},
	    {"t_time", "'T'HH:mm:ss.SSSZZ"},
	    {"strict_t_time", "'T'HH:mm:ss.SSSZZ"},
	    {"t_time_no_millis", "'T'HH:mm:ssZZ"},
	    {"strict_t_time_no_millis", "'T'HH:mm:ssZZ"},
	    {"week_date", "xxxx-'W'ww-e"},
	    {"strict_week_date", "xxxx-'W'ww-e"},
	    {"week_date_time", "xxxx-'W'ww-e'T'HH:mm:ss.SSSZZ"},
	    {"strict_week_date_time", "xxxx-'W'ww-e'T'HH:mm:ss.SSSZZ"},
	    {"week_date_time_no_millis", "xxxx-'W'ww-e'T'HH:mm:ssZZ"},
	    {"strict_week_date_time_no_millis", "xxxx-'W'ww-e'T'HH:mm:ssZZ"},
	    {"weekyear", "xxxx"},
	    {"strict_weekyear", "xxxx"},
	    {"weekyear_week", "xxxx-'W'ww"},
	    {"strict_weekyear_week", "xxxx-'W'ww"},
	    {"weekyear_week_day", "xxxx-'W'ww-e"},
	    {"strict_weekyear_week_day", "xxxx-'W'ww-e"},
	    {"year", "yyyy"},
	    {"strict_year", "yyyy"},
	    {"year_month", "yyyy-MM"},
	    {"strict_year_month", "yyyy-MM"},
	    {"year_month_day", "yyyy-MM-dd"},
	    {"strict_year_month_day", "yyyy-MM-dd"}};

	std::vector<std::string> time_col_names;

	for (const auto& col : cols) {
		if (col.second.first == "time") {
			time_col_names.emplace_back(col.first);
		}
	};

	if (time_col_names.empty()) {
		return;
	}

	// Get date formats from mapping
	std::vector<std::vector<std::string>> time_col_formats;
	{
		std::string json_buffer;
		std::string cols_names = boost::algorithm::join(time_col_names, ",");
		std::string url = socket() + "/" + _infos.get_index().toStdString() + "/_mapping/field/" +
		                  cols_names + "?include_defaults=true";

		prepare_query(_curl, url);
		if (perform_query(_curl, json_buffer)) {
			rapidjson::Document json;
			json.Parse<0>(json_buffer.c_str());

			rapidjson::Value& dates = json[_infos.get_index().toStdString().c_str()]["mappings"];
			if (not _mapping_type.empty()) {
				// for retro-compatibility purposes :
				// see
				// https://www.elastic.co/guide/en/elasticsearch/reference/current/removal-of-types.html
				dates = dates[_mapping_type.c_str()];
			}
			for (const std::string& col : time_col_names) {
				std::vector<std::string> col_name_hierarchy;
				boost::split(col_name_hierarchy, col, boost::is_any_of("."));

				const std::string& format =
				    dates[col.c_str()]["mapping"][col_name_hierarchy.back().c_str()]["format"]
				        .GetString();

				std::vector<std::string> subformats;
				boost::split(subformats, format, boost::is_any_of("||"));
				subformats.erase(std::remove_if(subformats.begin(), subformats.end(),
				                                [](const std::string& s) { return s.empty(); }));

				time_col_formats.emplace_back(subformats.empty() ? std::vector<std::string>{format} : std::move(subformats));
			}
		}
	}

	// get one result search filtered on time columns
	for (size_t i = 0; i < time_col_names.size(); i++) {
		std::string json_buffer;
		const std::string& col = time_col_names[i];
		std::string filter_path = get_filter_path_from_base(_curl, col, "hits.hits._source");
		std::string url = socket() + "/" + _infos.get_index().toStdString() +
		                  "/_search?size=1&filter_path=" + filter_path;

		// Filter-out null/empty values
		rapidjson::Document filter;
		rapidjson::Pointer("/query/constant_score/filter/exists/field").Set(filter, col.c_str());
		rapidjson::StringBuffer body;
		rapidjson::Writer<rapidjson::StringBuffer> writer(body);
		filter.Accept(writer);

		prepare_query(_curl, url, body.GetString());
		if (perform_query(_curl, json_buffer)) {
			rapidjson::Document json;
			json.Parse<0>(json_buffer.c_str());

			std::string params;
			if (json.GetObj().MemberCount() > 0) {
				std::string col_name_pointer = col;
				PVCore::replace(col_name_pointer, ".", "/");
				const rapidjson::Value* time =
				    rapidjson::Pointer(
				        (std::string("/hits/hits/0/_source/") + col_name_pointer).c_str())
				        .Get(json);
				if (time) {
					std::string time_value;
					if (time->IsUint64()) { // "epoch_second" and "epoch_millis" are stored as numbers
						time_value = std::to_string(time->IsUint64());
					}
					else {
						time_value = time->GetString();
					}
					for (const std::string& mapping_format : time_col_formats[i]) {
						if (not params.empty()) {
							break;
						}
						auto parse_date = [&time_value](const std::string& format) -> std::string {
							const auto& fd = PVRush::PVFormat::get_datetime_formatter_desc(format);
							pvcop::types::formatter_interface* fi =
							    pvcop::types::factory::create(fd.name(), fd.parameters());
							pvcop::db::array out_array(fi->name(), 1);
							if (fi->from_string(time_value.c_str(), out_array.data(), 0)) {
								return format;
							}
							return {};
						};

						
						auto format_str_it = dateformat_map.equal_range(mapping_format);
						for (auto it = format_str_it.first; it != format_str_it.second; ++it) {
							const std::string& format = it->second;
							params = parse_date(format);
							if (not params.empty()) {
								break;
							}
						}
						if (format_str_it.first == dateformat_map.end()) { // not a built-in format
							params = parse_date(mapping_format);
						}
					}
				}
			}

			auto it = std::find_if(cols.begin(), cols.end(),
			                       [&](const auto& c) { return c.first == col; });

			if (params.empty()) {
				it->second.first = "string";
			} else {
				it->second.second = params;
			}
		}
	}
}

QDomDocument PVRush::PVElasticsearchAPI::get_format_from_mapping() const
{
	QDomDocument format_doc;
	std::unique_ptr<PVXmlTreeNodeDom> format_root(PVRush::PVXmlTreeNodeDom::new_format(format_doc));

	for (const auto& col : format_columns(_infos.get_filter_path().toStdString())) {
		const std::string& column_name = col.first;
		const std::string& column_type = col.second.first;
		const std::string& column_format = col.second.second;

		PVRush::PVXmlTreeNodeDom* node = format_root->addOneField(
		    QString::fromStdString(column_name), QString::fromStdString(column_type));
		if (column_type == "time") {
			node->setAttribute(QString(PVFORMAT_AXIS_TYPE_FORMAT_STR), column_format.c_str());
		}
	}

	return format_doc;
}

size_t PVRush::PVElasticsearchAPI::count(const PVRush::PVElasticsearchQuery& query,
                                         std::string* error /* = nullptr */) const
{
	const PVElasticsearchInfos& infos = query.get_infos();
	std::string json_buffer;
	std::string url = socket() + "/" + infos.get_index().toStdString() + "/_count";

	prepare_query(_curl, url, query.get_query().toStdString());
	if (perform_query(_curl, json_buffer, error)) {

		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (_version < PVCore::PVVersion(5, 0, 0)) {
			rapidjson::Value& json_shards = json["_shards"];
			if (json_shards.HasMember("failures")) {
				if (error) {
					*error = json_shards["failures"][0]["reason"].GetString();
				}
				return 0;
			}
		} else if (has_error(json, error)) {
			return 0;
		}

		return json["count"].GetUint();
	}

	return 0;
}

bool PVRush::PVElasticsearchAPI::extract(const PVRush::PVElasticsearchQuery& query,
                                         PVRush::PVElasticsearchAPI::rows_t& rows,
                                         std::string* error /* = nullptr */)
{
	if (_init_scroll) {
		_slice_count = (_version >= PVCore::PVVersion(5, 0, 0))
		                   ? shards_count(query.get_infos().get_index().toStdString())
		                   : 1;
		_max_result_window = max_result_window(query.get_infos().get_index().toStdString());

		_curls.resize(_slice_count);
		_scroll_ids.resize(_slice_count);
		for (size_t i = 0; i < _slice_count; i++) {
			_curls[i] = curlp_t(curl_easy_init(), [](CURL* c) { curl_easy_cleanup(c); });
		}
	}

	std::vector<rows_t> local_rows(_slice_count);

	std::atomic<bool> end(false);
#pragma omp parallel for schedule(static, 1)
	for (size_t i = 0; i < _slice_count; i++) {
		CURL* curl = _curls[i].get();
		std::string json_buffer;

		scroll(curl, query, _init_scroll, i, _slice_count, _max_result_window, json_buffer, error);

		end = end || parse_scroll_results(curl, json_buffer, _scroll_ids[i], local_rows[i]);
	}
	_init_scroll = false;

	// merge rows
	rows.reserve(
	    std::accumulate(local_rows.begin(), local_rows.end(), 0,
	                    [](auto size, auto const& local_row) { return size + local_row.size(); }));
	for (auto& local_row : local_rows) {
		rows.insert(rows.end(), std::make_move_iterator(local_row.begin()),
		            std::make_move_iterator(local_row.end()));
	}

	return not end;
}

size_t PVRush::PVElasticsearchAPI::scroll_count() const
{
	return _scroll_count;
}

bool PVRush::PVElasticsearchAPI::clear_scroll()
{
	std::string result;
	std::string url = socket() + "/_search/scroll";

	curl_easy_setopt(_curl, CURLOPT_CUSTOMREQUEST, "DELETE");

	if (_version < PVCore::PVVersion(2, 0, 0)) {
		prepare_query(_curl, url, _scroll_ids[0]);
	} else {
		rapidjson::Document json;
		json.SetObject();

		rapidjson::Value scroll_ids(rapidjson::kArrayType);
		for (auto & _scroll_id : _scroll_ids) {
			rapidjson::Value sid(_scroll_id.c_str(), json.GetAllocator());
			scroll_ids.PushBack(sid, json.GetAllocator());
		}

		json.AddMember("scroll_id", scroll_ids, json.GetAllocator());

		rapidjson::StringBuffer strbuf;
		rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
		json.Accept(writer);

		prepare_query(_curl, url, strbuf.GetString());
	}

	bool res = perform_query(_curl, result);

	curl_easy_setopt(_curl, CURLOPT_CUSTOMREQUEST, nullptr);

	_curls.clear();
	_scroll_ids.clear();
	_init_scroll = true;

	return res;
}

std::string PVRush::PVElasticsearchAPI::sql_to_json(const std::string& sql,
                                                    std::string* error /* = nullptr */) const
{
	std::string json_buffer;
	std::string url;

	// URL-encode SQL request
	char* sql_url_encoded = curl_easy_escape(_curl, sql.c_str(), sql.size());
	if (sql_url_encoded == nullptr) {
		if (error) {
			*error = "Unable to URL-encode SQL query";
		}
		return {};
	}

	url = socket() + "/_sql/_explain?sql=" + sql_url_encoded;
	curl_free(sql_url_encoded);

	prepare_query(_curl, url);
	if (perform_query(_curl, json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (has_error(json, error)) {
			return {};
		}

		rapidjson::StringBuffer strbuf;
		rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
		json["query"].Accept(writer);

		std::stringstream query;

		query << "{ \"query\" : " << strbuf.GetString() << "}";

		return query.str();
	}

	return {};
}

bool PVRush::PVElasticsearchAPI::is_sql_available() const
{
	std::string json_buffer;
	std::string url = socket() + "/_sql";

	prepare_query(_curl, url);

	if (perform_query(_curl, json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (json.HasMember("status")) {
			size_t status_code = json["status"].GetUint();
			return status_code == 500;
		}
	}

	return false;
}

bool PVRush::PVElasticsearchAPI::init_scroll(CURL* curl,
                                             const PVRush::PVElasticsearchQuery& query,
                                             const uint64_t slice_id,
                                             const uint64_t slice_count,
                                             const uint64_t max_result_window,
                                             std::string& json_buffer,
                                             std::string* error /* = nullptr */)
{
	const PVElasticsearchInfos& infos = query.get_infos();
	bool res = false;
	bool disable_filter_path = false;
	do {
		const std::string& filter_path_var =
		    "&filter_path=_scroll_id,hits.total," +
		    get_filter_path_from_base(curl, infos.get_filter_path().toStdString(),
		                              "hits.hits._source");
		std::string url = socket() + "/" + infos.get_index().toStdString() +
		                  "/_search?scroll=" + SCROLL_TIMEOUT +
		                  (not disable_filter_path ? filter_path_var : "");
		if (_version < PVCore::PVVersion(5, 0, 0)) {
			url += "&search_type=scan";
		}

		rapidjson::Document json;
		json.Parse<0>(query.get_query().toStdString().c_str());

		if (slice_count > 1) {
			rapidjson::Value slice;
			slice.SetObject();
			slice.AddMember("id", slice_id, json.GetAllocator());
			slice.AddMember("max", slice_count, json.GetAllocator());
			json.AddMember("slice", slice, json.GetAllocator());
			/*
			"slice": {
			    "id": 0,
			    "max": 12
			}
			*/

			rapidjson::Value sort;
			sort.SetArray();
			sort.PushBack("_doc", json.GetAllocator());
			json.AddMember("sort", sort, json.GetAllocator());
			/*
			"sort": [
			    "_doc"
			]
			*/
		}

		json.AddMember("size", max_result_window, json.GetAllocator());

		// Source filtering
		std::vector<std::string> columns;
		const std::string& filter_path = infos.get_filter_path().toStdString();
		boost::algorithm::split(columns, filter_path, boost::is_any_of(","));
		rapidjson::Value source;
		source.SetArray();
		for (const std::string& column : columns) {
			rapidjson::Value col;
			col.SetString(column.c_str(), json.GetAllocator());
			source.PushBack(col, json.GetAllocator());
		}
		json.AddMember("_source", source, json.GetAllocator());

		rapidjson::StringBuffer strbuf;
		rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
		json.Accept(writer);

		prepare_query(curl, url, strbuf.GetString());

		perform_query(curl, json_buffer, error);

		rapidjson::Document json_scroll;
		json_scroll.Parse<0>(json_buffer.c_str());
		const rapidjson::Value* error_type =
		    rapidjson::Pointer("/error/root_cause/0/type").Get(json_scroll);
		if (error_type and std::string(error_type->GetString()) == "too_long_frame_exception") {
			// disable filter_path and retry
			disable_filter_path = true;
			json_buffer.clear();
			pvlogger::warn() << "Disabling 'filter_path' for this request as it caused "
			                    "'too_long_frame_exception' error"
			                 << std::endl;
		} else {
			res = true;
		}
	} while (res == false);

	return true;
}

void PVRush::PVElasticsearchAPI::update_scroll_id(CURL* curl, const std::string& scroll_id) const
{
	std::string url = socket() + "/_search/scroll";

	rapidjson::Document json;
	json.SetObject();
	json.AddMember("scroll", SCROLL_TIMEOUT, json.GetAllocator());
	rapidjson::Value sid(scroll_id.c_str(), json.GetAllocator());
	json.AddMember("scroll_id", sid, json.GetAllocator());

	rapidjson::StringBuffer strbuf;
	rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
	json.Accept(writer);

	prepare_query(curl, url, strbuf.GetString());
}

bool PVRush::PVElasticsearchAPI::scroll(CURL* curl,
                                        const PVRush::PVElasticsearchQuery& query,
                                        bool init,
                                        const size_t slice_id,
                                        const size_t slice_count,
                                        const size_t max_result_window,
                                        std::string& json_buffer,
                                        std::string* error /* = nullptr */
)
{
	if (init) {
		return init_scroll(curl, query, slice_id, slice_count, max_result_window, json_buffer,
		                   error);
	} else {
		return perform_query(curl, json_buffer, error);
	}
}

bool PVRush::PVElasticsearchAPI::parse_scroll_results(CURL* curl,
                                                      const std::string& json_data,
                                                      std::string& scroll_id,
                                                      rows_t& rows)
{
	rapidjson::Reader reader;
	PVElasticsearchSAXParser parser(_infos.get_filter_path().toStdString(), rows);
	rapidjson::StringStream ss(json_data.c_str());
	reader.Parse(ss, parser);

	scroll_id = parser.scroll_id();

	update_scroll_id(curl, scroll_id);
	_scroll_count = parser.total();

	return parser.end();
}

void PVRush::PVElasticsearchAPI::prepare_query(CURL* curl,
                                               const std::string& uri,
                                               const std::string& body /* = std::string() */
                                               ) const
{
	curl_easy_setopt(curl, CURLOPT_URL, uri.c_str());
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
	curl_easy_setopt(curl, CURLOPT_POST, not body.empty());

	if (not body.empty()) {
		curl_easy_setopt(curl, CURLOPT_COPYPOSTFIELDS, body.c_str());
	}
	if (not _infos.get_login().isEmpty()) {
		curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_ANY);
		curl_easy_setopt(
		    curl, CURLOPT_USERPWD,
		    (_infos.get_login().toStdString() + ":" + _infos.get_password().toStdString()).c_str());
	}
	curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, false);
	curl_easy_setopt(curl, CURLOPT_NOSIGNAL, true);

	// Set Content-Type header to "application/json"
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, _curl_headers);
}

bool PVRush::PVElasticsearchAPI::perform_query(CURL* curl,
                                               std::string& result,
                                               std::string* error /* = nullptr */) const
{
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);
	CURLcode res = curl_easy_perform(curl);

	if (res != CURLE_OK) {
		if (error) {
			*error = curl_easy_strerror(res);
		}
		return false;
	}

	return true;
}

std::string PVRush::PVElasticsearchAPI::socket() const
{
	std::stringstream socket;
	socket << _infos.get_host().toStdString() << ":" << _infos.get_port();

	return socket.str();
}
