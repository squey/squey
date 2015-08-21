/**
 * @file PVElasticsearchAPI.cpp
 *
 * @copyright (C) Picviz Labs 2015
 */

#include "PVElasticsearchAPI.h"
#include "PVElasticsearchInfos.h"
#include "PVElasticsearchQuery.h"

#include <sstream>
#include <thread>
#include <unordered_map>

#include <curl/curl.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include <tbb/pipeline.h>

#define USE_DOM_API 1

static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);

    return size * nmemb;
}

class json_parser : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, json_parser>
{
public:
    json_parser()
	{
	}

    bool StartObject()
    {
		level++;
		return true;
    }

    bool EndObject(rapidjson::SizeType)
    {
    	level--;
    	return true;
    }

    bool String(const char* s, rapidjson::SizeType length, bool)
    {
    	std::string str(std::move(s), length);
    	if (level==4 && str != "message") {
    		rows.emplace_back(str);
    	}
        return true;
    }

    PVRush::PVElasticsearchAPI::rows_t rows;

private:
    size_t level = 0;
};

PVRush::PVElasticsearchAPI::PVElasticsearchAPI(const PVRush::PVElasticsearchInfos& infos) : _curl(nullptr), _infos(infos)
{
	_curl = curl_easy_init();
}

PVRush::PVElasticsearchAPI::~PVElasticsearchAPI()
{
	curl_easy_cleanup(_curl);
}

void PVRush::PVElasticsearchAPI::prepare_query(const std::string& uri, const std::string& body /* = std::string() */) const
{
	curl_easy_setopt(_curl, CURLOPT_URL, uri.c_str());
	curl_easy_setopt(_curl, CURLOPT_WRITEFUNCTION, write_callback);
	if (body.empty() == false) {
		curl_easy_setopt(_curl, CURLOPT_POSTFIELDS, body.c_str());
	}
	if (_infos.get_login().isEmpty() == false) {
		curl_easy_setopt(_curl, CURLOPT_HTTPAUTH, CURLAUTH_ANY);
		curl_easy_setopt(_curl, CURLOPT_USERPWD, (_infos.get_login().toStdString() + ":" + _infos.get_password().toStdString()).c_str());
	}
	curl_easy_setopt(_curl, CURLOPT_SSL_VERIFYPEER, false);
}

bool PVRush::PVElasticsearchAPI::perform_query(std::string& result, std::string* error /* = nullptr */) const
{
	curl_easy_setopt(_curl, CURLOPT_WRITEDATA, &result);
	CURLcode res = curl_easy_perform(_curl);

	if(res != CURLE_OK) {
		if (error) {
			*error = curl_easy_strerror(res);
		}
		return false;
	}

	return true;
}

bool PVRush::PVElasticsearchAPI::is_sql_available() const
{
	std::string json_buffer;
	std::stringstream url;
	url << socket() << "/_sql";

	prepare_query(url.str());

	if (perform_query(json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (json.HasMember("status")) {
			size_t status_code = json["status"].GetUint();
			return status_code != 400;
		}
	}

	return false;
}

bool PVRush::PVElasticsearchAPI::check_connection(std::string* error /* =  nullptr */) const
{
	std::string json_buffer;
	std::stringstream url;
	url << socket();

	prepare_query(url.str());

	return perform_query(json_buffer, error);
}

std::string PVRush::PVElasticsearchAPI::sql_to_json(const std::string& sql, std::string* error /* = nullptr */) const
{
	std::string json_buffer;
	std::stringstream url;

	// URL-encode SQL request
	char* sql_url_encoded = curl_easy_escape(_curl, sql.c_str(), sql.size());
	if(sql_url_encoded) {
		url << socket() << "/_sql/_explain?sql=" << sql_url_encoded;
		curl_free(sql_url_encoded);
	}
	else {
		return std::string();
	}

	prepare_query(url.str());
	if (perform_query(json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (json.HasMember("error")) {
			if (error) {
				*error =  json["error"].GetString();
			}
			return std::string();
		}

		rapidjson::StringBuffer strbuf;
		rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
		json["query"].Accept(writer);

		std::stringstream query;

		query << "\"query\" : " << strbuf.GetString();

		return query.str();
	}

	return std::string();
}

std::string PVRush::PVElasticsearchAPI::socket() const
{
	std::stringstream socket;
	socket << _infos.get_host().toStdString() << ":" << _infos.get_port();

	return socket.str();
}

PVRush::PVElasticsearchAPI::indexes_t PVRush::PVElasticsearchAPI::indexes() const
{
	indexes_t indexes;
	std::string json_buffer;
	std::stringstream url;
	url << socket() << "/_stats";

	prepare_query(url.str());
	if (perform_query(json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (json.HasMember("error")) {
			PVLOG_ERROR("Error: %s\n", json["error"].GetString());
			return indexes_t();
		}

		rapidjson::Value& json_indexes = json["indices"];

		for (rapidjson::Value::ConstMemberIterator itr = json_indexes.MemberBegin(); itr != json_indexes.MemberEnd(); ++itr) {
			indexes.push_back(itr->name.GetString());
		}
	}

	return indexes;
}

PVRush::PVElasticsearchAPI::columns_t PVRush::PVElasticsearchAPI::columns(const PVRush::PVElasticsearchQuery& query) const
{
	columns_t cols;

	const PVElasticsearchInfos& infos = query.get_infos();

	if (infos.get_index().isEmpty()) {
		if (error) {
			*error =  "No index specified";
		}
		return cols;
	}

	std::string json_buffer;
	std::stringstream url;
	url << socket() << "/" << infos.get_index().toStdString() << "/_mapping";

	prepare_query(url.str());
	if (perform_query(json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (json.HasMember("error")) {
			PVLOG_ERROR("Error: %s\n", json["error"].GetString());
			return cols;
		}

		// TODO: if importer == "Logstash"

		rapidjson::Value& json_mappings = json[infos.get_index().toStdString().c_str()]["mappings"];

		// Several mappings can potentially be defined but we can only chose one...
		std::string mapping_type = "_default_";
		for (rapidjson::Value::ConstMemberIterator mtype = json_mappings.MemberBegin(); mtype != json_mappings.MemberEnd(); ++mtype) {
			mapping_type = mtype->name.GetString();
			if (mapping_type.size() > 0 && mapping_type[0] != '_') {
				break;
			}
		}

		rapidjson::Value& json_axes = json_mappings[mapping_type.c_str()]["properties"];

		const std::vector<std::string> invalid_cols = { "message", "type", "host", "path", "geoip" };
		std::unordered_map<std::string, std::string> types_mapping = { { "long", "integer" }, { "float", "double" } };

		for (rapidjson::Value::ConstMemberIterator axe = json_axes.MemberBegin(); axe != json_axes.MemberEnd(); ++axe) {
			std::string name = axe->name.GetString();

			if (std::find(invalid_cols.begin(), invalid_cols.end(), name) == invalid_cols.end() && (name.size() > 0 && name[0] != '@')) {
				std::string type = json_axes[name.c_str()]["type"].GetString();

				auto it = types_mapping.find(type);
				if (it != types_mapping.end()) {
					type = it->second;
				}

				cols.emplace_back(name, type);
			}
		}
	}

	return cols;
}

size_t PVRush::PVElasticsearchAPI::count(const PVRush::PVElasticsearchQuery& query) const
{
	const PVElasticsearchInfos& infos = query.get_infos();
	std::string json_buffer;
	std::stringstream url;
	url << socket() << "/" << infos.get_index().toStdString() << "/_count";

	std::string complete_request = std::string("{") + query.get_query().toStdString() + std::string("}");

	prepare_query(url.str(), complete_request);
	if (perform_query(json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (json.HasMember("error")) {
			PVLOG_ERROR("Error: %s\n", json["error"].GetString());
			return 0;
		}

		return json["count"].GetUint();
	}

	return 0;
}

bool PVRush::PVElasticsearchAPI::init_scroll(const PVRush::PVElasticsearchQuery& query)
{
	const PVElasticsearchInfos& infos = query.get_infos();
	std::string json_buffer;
	std::stringstream url;
	url << socket() << "/" << infos.get_index().toStdString() << "/_search?scroll=1m&search_type=scan";

	// TODO: test "_source : message" instead of "fields : message" to see if there is any performance change (don't forget to update the parsing)
	std::string complete_request = std::string("{ \
		\"fields\": \"message\", \
		\"size\":  1000," \
		+ query.get_query().toStdString() + std::string("}")
	).c_str();

	prepare_query(url.str(), complete_request);
	if (perform_query(json_buffer)) {
		rapidjson::Document json;
		json.Parse<0>(json_buffer.c_str());

		if (json.HasMember("error")) {
			PVLOG_ERROR("Error: %s\n", json["error"].GetString());
			return false;
		}

		_scroll_id = json["_scroll_id"].GetString();
		_scroll_count = json["hits"]["total"].GetUint();
	}

	url.clear();
	url.str(std::string());
	url << socket() << "/_search/scroll?scroll=1m";

	prepare_query(url.str(), _scroll_id);

	return true;
}

bool PVRush::PVElasticsearchAPI::scroll(const PVRush::PVElasticsearchQuery& query, std::string& json_buffer)
{
	if (_scroll_id.empty()) {
		init_scroll(query);
	}

	return perform_query(json_buffer);
}

size_t PVRush::PVElasticsearchAPI::scroll_count() const
{
	return _scroll_count;
}

bool PVRush::PVElasticsearchAPI::clear_scroll()
{
	std::string result;
	std::stringstream url;
	url << socket() << "/_search/scroll";
	curl_easy_setopt(_curl, CURLOPT_CUSTOMREQUEST, "DELETE");

	prepare_query(url.str(), _scroll_id);
	bool res = perform_query(result);

	curl_easy_setopt(_curl, CURLOPT_CUSTOMREQUEST, NULL);

	return res;
}

bool PVRush::PVElasticsearchAPI::parse_results(const std::string& json_data, rows_t& rows) const
{
#if USE_DOM_API
	rapidjson::Document json;
	json.Parse<0>(json_data.c_str());

	rapidjson::Value& hits = json["hits"]["hits"];

	rows.clear();
	rows.reserve(hits.Size());

	if (hits.Size() == 0) { // end of query
		return false;
	}

	for (rapidjson::SizeType i = 0; i < hits.Size(); i++) {
		const rapidjson::Value& message = hits[i]["fields"]["message"][0];
		rows.push_back(message.GetString());
	}

	return true;
#else // USE_SAX_API
	std::string csv_buffer;

	rapidjson::Reader reader;
    json_parser parser;
    rapidjson::StringStream ss(json_data.c_str());
    reader.Parse(ss, parser);
	rows = std::move(parser.rows);

    return rows.size() > 0;
#endif
}

bool PVRush::PVElasticsearchAPI::extract(const PVRush::PVElasticsearchQuery& query, PVRush::PVElasticsearchAPI::rows_chunk_t& rows_array) const
{
	int request_count = std::thread::hardware_concurrency();

	using indexed_json_buffer_t = std::pair<std::string, size_t>;

	rows_array.resize(request_count);

	bool query_end = false;

	tbb::parallel_pipeline(request_count,
		tbb::make_filter<void, indexed_json_buffer_t>(tbb::filter::serial_in_order,
			[&](tbb::flow_control& fc)
			{
				if (--request_count == -1 || query_end) {
					fc.stop();
					return indexed_json_buffer_t();
				}
				std::string json_buffer;

				scroll(query, json_buffer);

				return indexed_json_buffer_t(std::move(json_buffer), request_count);
			}
		) &
		tbb::make_filter<indexed_json_buffer_t, void>(tbb::filter::parallel,
			[&](indexed_json_buffer_t json_buffer)
			{
				if(parse_results(json_buffer.first, rows_array[json_buffer.second]) == false) {
					query_end = true;
				}
			}
		)
	);

	return query_end;
}
