/**
 * @file PVElasticsearchAPI.cpp
 *
 * @copyright (C) Picviz Labs 2015
 */

#include "PVElasticsearchAPI.h"
#include "PVElasticsearchInfos.h"
#include "PVElasticsearchQuery.h"

#include <sstream>

#include <curl/curl.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

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

bool PVRush::PVElasticsearchAPI::perform_query(std::string& result, std::string* error /* = nulltr */) const
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

std::string PVRush::PVElasticsearchAPI::sql_to_json(const std::string& sql) const
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

