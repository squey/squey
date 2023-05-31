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

#include "PVSplunkAPI.h"

#include "PVSplunkInfos.h"
#include "PVSplunkQuery.h"
#include "PVSplunkJsonConverter.h"

#include <QDomDocument>
#include <QByteArray>

#include <sstream>

#include <curl/curl.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

static constexpr size_t BUFFER_SIZE = 2 * 1024 * 1024;

static int socket_callback(
    CURL* /*easy*/, curl_socket_t s, int what, PVRush::PVSplunkAPI* splunk, void* /*sockdata*/
    )
{
	if (what == CURL_POLL_REMOVE) {
		splunk->set_socket(-1);
	} else if (what == CURL_POLL_IN) {
		splunk->set_socket(s);
	}

	return 0;
}

static int timer_callback(CURLM* /*multi*/, long timeout_ms, PVRush::PVSplunkAPI* splunk)
{
	splunk->set_timeout(timeout_ms);

	return 0;
}

static size_t
write_callback_export(void* data, size_t size, size_t count, PVRush::PVSplunkAPI* splunk)
{
	size_t real_size = size * count;

	splunk->append_data(data, real_size);

	return real_size;
}

static size_t write_callback_query(void* contents, size_t size, size_t nmemb, void* userp)
{
	((std::string*)userp)->append((char*)contents, size * nmemb);

	return size * nmemb;
}

PVRush::PVSplunkAPI::PVSplunkAPI(const PVRush::PVSplunkInfos& infos) : _infos(infos), _socket(-1)
{
	_multi = curl_multi_init();
	_easy = curl_easy_init();
}

PVRush::PVSplunkAPI::~PVSplunkAPI()
{
	if (_easy) {
		curl_multi_remove_handle(_multi, _easy);
		curl_easy_cleanup(_easy);
	}

	if (_multi) {
		curl_multi_cleanup(_multi);
	}
}

bool PVRush::PVSplunkAPI::check_connection(std::string* error /* =  nullptr */) const
{
	std::string buffer;

	if (not perform_query("", buffer, "json", error)) {
		return false; // Abort as the query didn't succeed
	}

	rapidjson::Document json;
	json.Parse<0>(buffer.c_str());

	if (not json.HasMember("messages")) {
		return false; // Abort as the answer is ill-formed
	}

	// We are connected if the server answer with a message type : FATAL
	// and message content == "Empty search."

	// Check message type
	rapidjson::Value& messages = json["messages"][0];
	bool correct_connection = std::string(messages["type"].GetString()) == "FATAL";
	// Check message content
	rapidjson::Value& error_msg = messages["text"];
	std::string err = error_msg.GetString();
	correct_connection &= err == "Empty search.";

	if (error and not correct_connection) {
		*error = err;
	}

	return correct_connection;
}

PVRush::PVSplunkAPI::strings_t
PVRush::PVSplunkAPI::indexes(std::string* error /* =  nullptr */) const
{
	return list("| eventcount summarize=f index=*", "index", error);
}

PVRush::PVSplunkAPI::strings_t PVRush::PVSplunkAPI::hosts(std::string* error /* =  nullptr */) const
{
	return list("metadata type=hosts", "host", error);
}

PVRush::PVSplunkAPI::strings_t
PVRush::PVSplunkAPI::sourcetypes(std::string* error /* =  nullptr */) const
{
	return list("metadata type=sourcetypes", "sourcetype", error);
}

PVRush::PVSplunkAPI::strings_t PVRush::PVSplunkAPI::list(const std::string& search,
                                                         const std::string& type,
                                                         std::string* error /* =  nullptr */) const
{
	PVRush::PVSplunkAPI::strings_t list;
	std::string buffer;

	if (perform_query(search, buffer, "json", error)) {
		std::istringstream iss(buffer);
		for (std::string line; std::getline(iss, line);) {
			rapidjson::Document json;
			json.Parse<0>(line.c_str());
			list.emplace_back(json["result"][type.c_str()].GetString());
		}
	}

	return list;
}

PVRush::PVSplunkAPI::columns_t PVRush::PVSplunkAPI::columns(std::string* error /*= nullptr*/) const
{
	std::string buffer;
	columns_t cols;

	std::string search_query = filtered_search() + "| head 1 | fields *";

	// Retrieve columns name (XML)
	if (!perform_query(search_query, buffer, "xml", error)) {
		return {};
	}

	// As splunk is unable to return valid XML content, we're going to build our own valid XML !
	buffer.insert(buffer.find_first_of(">") + 1, "<splunk_sucks>").append("</splunk_sucks>");

	QDomDocument xml;
	QString error_msg;
	if (!xml.setContent(QString(buffer.c_str()), true, &error_msg)) {
		if (error) {
			*error = error_msg.toStdString();
		}
		return {};
	}

	QDomElement dom = xml.documentElement();
	QDomNode root = dom.firstChild();
	QDomNode results = root.firstChild();
	QDomNode meta = results.firstChild();
	QDomNodeList fields = meta.toElement().elementsByTagName("field");
	for (int i = 0; i < fields.count(); i++) {
		QString column = fields.at(i).firstChild().toText().data();
		cols.emplace_back(column.toStdString(), "");
	}

	// Retrieve columns type (JSON)
	static constexpr char SQUEY_PREFIX[] = "squey_isnum_";
	for (auto& col : cols) {
		const std::string& column_name = col.first;
		search_query += std::string("| eval ") + SQUEY_PREFIX + column_name + "=if(isnum(" +
		                column_name + R"(),"1","0") )";
	}
	buffer.clear();
	if (perform_query(search_query, buffer, "json", error)) {
		rapidjson::Document json;
		json.Parse<0>(buffer.c_str());
		rapidjson::Value& json_result = json["result"];
		for (auto& col : cols) {
			const std::string& column_name = col.first;
			std::string& column_type = col.second;
			std::string type_name = SQUEY_PREFIX + column_name;
			column_type =
			    json_result[type_name.c_str()].GetString()[0] == '1' ? "integer" : "string";
		}
	}

	return cols;
}

size_t PVRush::PVSplunkAPI::count(const PVRush::PVSplunkQuery& query,
                                  std::string* error /* = nullptr */) const
{
	std::string json_buffer;
	std::string search = filtered_search() + query.get_query().toStdString() + " | stats count";

	if (perform_query(search, json_buffer, "json", error)) {

		std::istringstream is(json_buffer.c_str());
		std::string line;
		std::string last_line;
		while (std::getline(is, line)) {
			last_line = line;
		}

		rapidjson::Document json;
		json.Parse<0>(last_line.c_str());

		return std::atoll(json["result"]["count"].GetString());
	}

	return 0;
}

bool PVRush::PVSplunkAPI::extract(const PVRush::PVSplunkQuery& query,
                                  std::string& data_batch,
                                  std::string* error /* = nullptr */)
{
	std::string search_query = filtered_search() + query.get_query().toStdString();

	if (!_ongoing_extract_query) {
		prepare_extract(search_query, error);
	}

	while (poll()) {
		if (extract_buffer_size() > BUFFER_SIZE) {
			extract_buffer(data_batch);
			return true;
		}
	};

	if (extract_buffer_size()) {
		extract_buffer(data_batch);
	}

	return false;
}

std::string PVRush::PVSplunkAPI::rules_to_json(const std::string& rules) const
{
	PVSplunkJsonConverter sjc(rules);
	return sjc.rules_to_json();
}

bool PVRush::PVSplunkAPI::poll()
{
	struct timeval tv = _tv;
	int ret;
	CURLMcode rc;

	if (_socket < 0) {
		usleep(10000);
		ret = 0;
	} else {
		FD_SET(_socket, &_fdset);
		ret = select(_socket + 1, &_fdset, nullptr, nullptr, &tv);
	}

	if (ret == 0) {
		rc = curl_multi_socket_action(_multi, CURL_SOCKET_TIMEOUT, 0, &_ongoing_extract_query);
	} else {
		rc = curl_multi_socket_action(_multi, _socket, CURL_CSELECT_IN, &_ongoing_extract_query);
	}

	if (rc != CURLM_OK) {
		return false;
	}
	return _ongoing_extract_query > 0;
}

void PVRush::PVSplunkAPI::extract_buffer(std::string& buffer)
{
	std::lock_guard<std::mutex> lock(_mutex);

	size_t pos = _data.find_last_of("\n");

	std::string last_line = _data.substr(pos + 1, _data.size());

	_data.resize(_data.size() - last_line.size());

	buffer = std::move(_data);

	_data = std::move(last_line);
}

void PVRush::PVSplunkAPI::prepare_extract(const std::string& search_query,
                                          std::string* /* error = nullptr */)
{
	_extract_canceled = false;
	_data.clear();

	std::string url = export_api_url(search_query, "raw");
	std::string login = _infos.get_login().toStdString();

	std::string credential;
	if (login.empty() == false) {
		credential += login;
		std::string passwd = _infos.get_password().toStdString();
		if (passwd.empty() == false) {
			credential += ":";
			credential += passwd;
		}
	}

	curl_multi_setopt(_multi, CURLMOPT_SOCKETFUNCTION, socket_callback);
	curl_multi_setopt(_multi, CURLMOPT_SOCKETDATA, this);

	curl_multi_setopt(_multi, CURLMOPT_TIMERFUNCTION, timer_callback);
	curl_multi_setopt(_multi, CURLMOPT_TIMERDATA, this);

	curl_multi_setopt(_multi, CURLMOPT_PIPELINING, 1L);

	curl_easy_setopt(_easy, CURLOPT_WRITEFUNCTION, write_callback_export);
	curl_easy_setopt(_easy, CURLOPT_WRITEDATA, this);
	curl_easy_setopt(_easy, CURLOPT_PRIVATE, this);

	// curl_easy_setopt(_easy, CURLOPT_PIPEWAIT, 1L); // Added in 7.43.0

	curl_easy_setopt(_easy, CURLOPT_HTTPAUTH, CURLAUTH_ANY);
	curl_easy_setopt(_easy, CURLOPT_USERPWD, credential.c_str());
	curl_easy_setopt(_easy, CURLOPT_SSL_VERIFYPEER, 0L);
	curl_easy_setopt(_easy, CURLOPT_SSL_VERIFYHOST, 0L);

	curl_easy_setopt(_easy, CURLOPT_URL, url.c_str());
	curl_easy_setopt(_easy, CURLOPT_HTTPGET, 1L);
	curl_easy_setopt(_easy, CURLOPT_MAXREDIRS, 50L);
	curl_easy_setopt(_easy, CURLOPT_TCP_KEEPALIVE, 1L);

	curl_multi_add_handle(_multi, _easy);

	FD_ZERO(&_fdset);
}

bool PVRush::PVSplunkAPI::perform_query(const std::string& search_query,
                                        std::string& result,
                                        const std::string& output_mode /* = "json" */,
                                        std::string* error /* = nullptr */
                                        ) const
{
	std::string url = export_api_url(search_query, output_mode);

	curl_easy_setopt(_easy, CURLOPT_URL, url.c_str());
	curl_easy_setopt(_easy, CURLOPT_WRITEFUNCTION, write_callback_query);
	curl_easy_setopt(_easy, CURLOPT_WRITEDATA, &result);
	if (_infos.get_login().isEmpty() == false) {
		curl_easy_setopt(_easy, CURLOPT_HTTPAUTH, CURLAUTH_ANY);
		curl_easy_setopt(
		    _easy, CURLOPT_USERPWD,
		    (_infos.get_login().toStdString() + ":" + _infos.get_password().toStdString()).c_str());
	}
	curl_easy_setopt(_easy, CURLOPT_SSL_VERIFYPEER, 0L);
	curl_easy_setopt(_easy, CURLOPT_SSL_VERIFYHOST, 0L);
	CURLcode res = curl_easy_perform(_easy);

	if (res != CURLE_OK) {
		if (error) {
			*error = curl_easy_strerror(res);
		}
		return false;
	}

	return true;
}

std::string PVRush::PVSplunkAPI::export_api_url(const std::string& search_query,
                                                const std::string& output_mode /* = "json" */,
                                                std::string* /* error  =  nullptr */
                                                ) const
{
	std::stringstream api_url;

	api_url << _infos.get_host().toStdString() << ":" << _infos.get_port();

	std::string login = _infos.get_login().toStdString();
	if (login.empty() == false) {
		api_url << "/servicesNS/" << login << "/search/search/jobs/export";
	} else {
		// TODO: find if an export can be run without auth
		api_url << "/services";
	}

	char* escaped_search_query = curl_easy_escape(_easy, search_query.c_str(), 0);

	api_url << "?search=" << escaped_search_query << "&output_mode=" << output_mode
	        << "&exec_mode=oneshot";

	curl_free(escaped_search_query);

	return api_url.str();
}

std::string PVRush::PVSplunkAPI::filtered_search() const
{
	std::string index = _infos.get_splunk_index().toStdString();
	std::string host = _infos.get_splunk_host().toStdString();
	std::string sourcetype = _infos.get_splunk_sourcetype().toStdString();

	std::string search = "search";
	if (index.empty() == false) {
		search += " index=" + index;
	}
	if (host.empty() == false) {
		search += " host=" + host;
	}
	if (sourcetype.empty() == false) {
		search += " sourcetype=" + sourcetype;
	}

	return search + " ";
}

void PVRush::PVSplunkAPI::set_socket(int s)
{
	_socket = s;
}

void PVRush::PVSplunkAPI::set_timeout(long t)
{
	_tv.tv_sec = t / 1000;
	_tv.tv_usec = (t % 1000) * 1000;
}

/**
 * _data must be protected by a mutex because the extract_buffer method may want
 * to access it at the same time from another thread.
 */
void PVRush::PVSplunkAPI::append_data(const void* data, size_t size)
{
	std::lock_guard<std::mutex> lock(_mutex);

	_data.append(static_cast<const char*>(data), size);
}
