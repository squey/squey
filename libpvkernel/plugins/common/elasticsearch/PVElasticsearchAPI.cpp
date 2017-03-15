/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include "PVElasticsearchAPI.h"
#include "PVElasticsearchInfos.h"
#include "PVElasticsearchQuery.h"
#include "PVElasticsearchJsonConverter.h"

#include <sstream>
#include <thread>
#include <unordered_map>
#include <atomic>

#include <curl/curl.h>

#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include <tbb/pipeline.h>

static constexpr const size_t SCROLL_SIZE = 1000;
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
	if (json.HasMember("error")) {
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

	_version = version();
}

PVRush::PVElasticsearchAPI::~PVElasticsearchAPI()
{
	curl_easy_cleanup(_curl);
}

bool PVRush::PVElasticsearchAPI::check_connection(std::string* error /* =  nullptr */) const
{
	std::string json_buffer;

	prepare_query(socket());

	perform_query(json_buffer, error);

	rapidjson::Document json;
	json.Parse<0>(json_buffer.c_str());
	return not has_error(json, error);
}

PVCore::PVVersion PVRush::PVElasticsearchAPI::version() const
{
	std::string json_buffer;
	std::string error;

	prepare_query(socket());
	perform_query(json_buffer, &error);

	if (error.empty()) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());
		if (json.IsObject() and json.HasMember("version") and json["version"].HasMember("number")) {
			return {json["version"]["number"].GetString()};
		}
	}

	// if we don't know the version, assume it's the last one
	return {(size_t)-1, (size_t)-1, (size_t)-1};
}

PVRush::PVElasticsearchAPI::indexes_t
PVRush::PVElasticsearchAPI::indexes(std::string* error /*= nullptr*/) const
{
	indexes_t indexes;
	std::string json_buffer;
	std::string url = socket() + "/_stats";

	prepare_query(url);
	if (perform_query(json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (has_error(json, error)) {
			return indexes_t();
		}

		rapidjson::Value& json_indexes = json["indices"];

		for (rapidjson::Value::ConstMemberIterator itr = json_indexes.MemberBegin();
		     itr != json_indexes.MemberEnd(); ++itr) {
			indexes.emplace_back(itr->name.GetString());
		}
	}

	return indexes;
}

PVRush::PVElasticsearchAPI::columns_t
PVRush::PVElasticsearchAPI::columns(const PVRush::PVElasticsearchQuery& query,
                                    std::string* error /*= nullptr*/) const
{
	columns_t cols;

	const PVElasticsearchInfos& infos = query.get_infos();

	if (infos.get_index().isEmpty()) {
		if (error) {
			*error = "No index specified";
		}
		return cols;
	}

	std::string json_buffer;
	std::string url = socket() + "/" + infos.get_index().toStdString() + "/_mapping";

	prepare_query(url);
	if (perform_query(json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (has_error(json, error)) {
			return cols;
		}

		rapidjson::Value& json_mappings = json[infos.get_index().toStdString().c_str()]["mappings"];

		// Several mappings can potentially be defined but we can only chose one...
		std::string mapping_type = "_default_";
		for (rapidjson::Value::ConstMemberIterator mtype = json_mappings.MemberBegin();
		     mtype != json_mappings.MemberEnd(); ++mtype) {
			mapping_type = mtype->name.GetString();
			if (mapping_type.size() > 0 && mapping_type[0] != '_') {
				break;
			}
		}

		rapidjson::Value& json_axes = json_mappings[mapping_type.c_str()]["properties"];

		static const std::vector<std::string> invalid_cols = {"message", "type", "host", "path",
		                                                      "geoip"};

		// type mapping between elasticsearch and querybuilder
		static const std::unordered_map<std::string, std::string> types_mapping = {
		    {"long", "integer"},  {"integer", "integer"}, {"short", "integer"},
		    {"byte", "integer"},  {"double", "double"},   {"float", "double"},
		    {"date", "datetime"}, {"text", "string"},     {"keyword", "string"}};
		auto map_type = [&](const std::string& type) -> std::string {
			const auto& it = types_mapping.find(type);
			if (it != types_mapping.end()) {
				return it->second;
			} else {
				// fallback type for unkown types
				return "string";
			}
		};

		for (rapidjson::Value::ConstMemberIterator axe = json_axes.MemberBegin();
		     axe != json_axes.MemberEnd(); ++axe) {
			std::string name = axe->name.GetString();

			if (std::find(invalid_cols.begin(), invalid_cols.end(), name) == invalid_cols.end() &&
			    (name.size() > 0 && name[0] != '@')) {
				const auto& field = json_axes[name.c_str()];
				if (field.HasMember("type")) {
					std::string type = field["type"].GetString();
					cols.emplace_back(name, map_type(type));
				} else if (field.HasMember("properties")) {
					const auto& properties = field["properties"];
					for (rapidjson::Value::ConstMemberIterator property = properties.MemberBegin();
					     property != properties.MemberEnd(); ++property) {
						std::string prop_name = property->name.GetString();
						if (properties[prop_name.c_str()].HasMember("type")) {
							std::string type = properties[prop_name.c_str()]["type"].GetString();
							cols.emplace_back(name + "." + prop_name, map_type(type));
						}
					}
				}
			}
		}
	}

	return cols;
}

size_t PVRush::PVElasticsearchAPI::count(const PVRush::PVElasticsearchQuery& query,
                                         std::string* error /* = nullptr */) const
{
	const PVElasticsearchInfos& infos = query.get_infos();
	std::string json_buffer;
	std::string url = socket() + "/" + infos.get_index().toStdString() + "/_count";

	prepare_query(url, query.get_query().toStdString());
	if (perform_query(json_buffer, error)) {

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

/* While the scroll API must be accessed sequentially
 * (otherwise the Elasticsearch cluster simply fails at some point)
 * the JSON content returned by the server can be parsed in parallel
 * using a pipeline.
 */
bool PVRush::PVElasticsearchAPI::extract(const PVRush::PVElasticsearchQuery& query,
                                         PVRush::PVElasticsearchAPI::rows_chunk_t& rows_array,
                                         std::string* error /* = nullptr */)
{
	int request_count = std::thread::hardware_concurrency();

	using indexed_json_buffer_t = std::pair<std::string, size_t>;

	rows_array.resize(request_count);

	std::atomic<bool> query_end;
	query_end = false;

	tbb::parallel_pipeline(
	    request_count,
	    tbb::make_filter<void, indexed_json_buffer_t>(
	        tbb::filter::serial_in_order,
	        [&](tbb::flow_control& fc) {
		        if (--request_count == -1 or query_end) {
			        fc.stop();
			        return indexed_json_buffer_t();
		        }
		        std::string json_buffer;

		        scroll(query, json_buffer, error);

		        if (error and not error->empty()) {
			        query_end = true;
		        }

		        return indexed_json_buffer_t(std::move(json_buffer), request_count);
		    }) &
	        tbb::make_filter<indexed_json_buffer_t, void>(
	            tbb::filter::parallel, [&](indexed_json_buffer_t json_buffer) {
		            if (parse_scroll_results(json_buffer.first, rows_array[json_buffer.second]) ==
		                false) {
			            query_end = true;
		            }
		        }));

	return query_end;
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
		prepare_query(url, _scroll_id);
	} else {
		rapidjson::Document json;
		json.SetObject();
		rapidjson::Value sid(_scroll_id.c_str(), json.GetAllocator());
		json.AddMember("scroll_id", sid, json.GetAllocator());

		rapidjson::StringBuffer strbuf;
		rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
		json.Accept(writer);

		prepare_query(url, strbuf.GetString());
	}

	bool res = perform_query(result);

	curl_easy_setopt(_curl, CURLOPT_CUSTOMREQUEST, nullptr);

	_scroll_id.clear();

	return res;
}

std::string PVRush::PVElasticsearchAPI::rules_to_json(const std::string& rules) const
{
	PVElasticSearchJsonConverter esc(_version, rules);
	return esc.rules_to_json();
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
		return std::string();
	}

	url = socket() + "/_sql/_explain?sql=" + sql_url_encoded;
	curl_free(sql_url_encoded);

	prepare_query(url);
	if (perform_query(json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (has_error(json, error)) {
			return std::string();
		}

		rapidjson::StringBuffer strbuf;
		rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
		json["query"].Accept(writer);

		std::stringstream query;

		query << "{ \"query\" : " << strbuf.GetString() << "}";

		return query.str();
	}

	return std::string();
}

bool PVRush::PVElasticsearchAPI::is_sql_available() const
{
	std::string json_buffer;
	std::string url = socket() + "/_sql";

	prepare_query(url);

	if (perform_query(json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (json.HasMember("status")) {
			size_t status_code = json["status"].GetUint();
			return status_code == 500;
		}
	}

	return false;
}

bool PVRush::PVElasticsearchAPI::init_scroll(const PVRush::PVElasticsearchQuery& query,
                                             std::string& json_buffer,
                                             std::string* error /* = nullptr */)
{
	const PVElasticsearchInfos& infos = query.get_infos();
	std::string url = socket() + "/" + infos.get_index().toStdString() +
	                  "/_search?filter_path=hits.total,hits.hits._source,_scroll_id&scroll=" +
	                  SCROLL_TIMEOUT;
	if (_version < PVCore::PVVersion(5, 0, 0)) {
		url += "&search_type=scan";
	}

	rapidjson::Document json;
	json.Parse<0>(query.get_query().toStdString().c_str());

	json.AddMember("_source", "message", json.GetAllocator());
	json.AddMember("size", SCROLL_SIZE, json.GetAllocator());

	rapidjson::StringBuffer strbuf;
	rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
	json.Accept(writer);

	prepare_query(url, strbuf.GetString());
	if (perform_query(json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (has_error(json, error)) {
			return false;
		}

		_scroll_id = json["_scroll_id"].GetString();
		_scroll_count = json["hits"]["total"].GetUint();
	}

	url.clear();
	url = socket() + "/_search/scroll?scroll";

#define SCROLL_API_POST 0 // should work according to the documentation, but doesn't

#if SCROLL_API_POST
	if (_version < PVCore::PVVersion(2, 0, 0)) {
#endif
		url += "=" + std::string(SCROLL_TIMEOUT);
		prepare_query(url, _scroll_id);
#if SCROLL_API_POST
	} else {
		rapidjson::Document json;
		json.SetObject();
		json.AddMember("scroll", SCROLL_TIMEOUT, json.GetAllocator());
		rapidjson::Value sid(_scroll_id.c_str(), json.GetAllocator());
		json.AddMember("scroll_id", sid, json.GetAllocator());

		rapidjson::StringBuffer strbuf;
		rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
		json.Accept(writer);

		prepare_query(url, strbuf.GetString());
	}
#endif

	return true;
}

bool PVRush::PVElasticsearchAPI::scroll(const PVRush::PVElasticsearchQuery& query,
                                        std::string& json_buffer,
                                        std::string* error /* = nullptr */
                                        )
{
	if (_scroll_id.empty()) {
		bool res = init_scroll(query, json_buffer, error);
		if (_version >= PVCore::PVVersion(5, 2, 0)) {
			return res;
		}
	}

	return perform_query(json_buffer, error);
}

bool PVRush::PVElasticsearchAPI::parse_scroll_results(const std::string& json_data,
                                                      rows_t& rows) const
{
	rapidjson::Document json;
	json.Parse<0>(json_data.c_str());
	std::string error;
	if (json_data.empty() or has_error(json, &error)) {
		return false;
	}

	rapidjson::Value& hits = json["hits"]["hits"];

	rows.clear();
	rows.reserve(hits.Size());

	if (hits.Size() == 0) { // end of query
		return false;
	}

	for (rapidjson::SizeType i = 0; i < hits.Size(); i++) {
		const rapidjson::Value& message = (_version < PVCore::PVVersion(5, 0, 0))
		                                      ? hits[i]["_source"]["message"][0]
		                                      : hits[i]["_source"]["message"];
		rows.emplace_back(message.GetString());
	}

	return true;
}

/**
 * Beware that curl_easy_setopt() *does not* make a string copy !
 */
void PVRush::PVElasticsearchAPI::prepare_query(const std::string& uri,
                                               const std::string& body /* = std::string() */) const
{
	curl_easy_setopt(_curl, CURLOPT_URL, uri.c_str());
	curl_easy_setopt(_curl, CURLOPT_WRITEFUNCTION, write_callback);
	curl_easy_setopt(_curl, CURLOPT_POST, not body.empty());
	if (not body.empty()) {
		curl_easy_setopt(_curl, CURLOPT_COPYPOSTFIELDS, body.c_str());
	}
	if (not _infos.get_login().isEmpty()) {
		curl_easy_setopt(_curl, CURLOPT_HTTPAUTH, CURLAUTH_ANY);
		curl_easy_setopt(
		    _curl, CURLOPT_USERPWD,
		    (_infos.get_login().toStdString() + ":" + _infos.get_password().toStdString()).c_str());
	}
	curl_easy_setopt(_curl, CURLOPT_SSL_VERIFYPEER, false);
}

bool PVRush::PVElasticsearchAPI::perform_query(std::string& result,
                                               std::string* error /* = nullptr */) const
{
	curl_easy_setopt(_curl, CURLOPT_WRITEDATA, &result);
	CURLcode res = curl_easy_perform(_curl);

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
