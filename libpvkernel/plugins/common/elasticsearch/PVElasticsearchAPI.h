/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVELASTICSEARCHAPI_H__
#define __PVELASTICSEARCHAPI_H__

#include <QDomDocument>

#include <vector>
#include <string>

#include "PVElasticsearchInfos.h"

#include <pvkernel/core/PVVersion.h>

#include <curl/curl.h>

#include <rapidjson/document.h>

namespace PVRush
{

class PVElasticsearchQuery;

/** This class contains all the logic needed by INENDI Inspector
 *  to communicate with an Elasticsearch server.
 */
class PVElasticsearchAPI
{
  public:
	static constexpr size_t DEFAULT_PORT = 9200;

	// type mapping between elasticsearch and inspector format
	static const std::unordered_map<std::string, std::string>& types_mapping()
	{
		static const std::unordered_map<std::string, std::string> types_mapping = {
		    {"long", "long"},
		    {"integer", "integer"},
		    {"short", "short"},
		    {"byte", "byte"},
		    {"double", "number_double"},
		    {"float", "number_float"},
		    {"half_float", "number_float"},
		    {"date", "time"},
		    {"ip", "ipv6"},
		    {"text", "string"},
		    {"keyword", "string"},
		    {"boolean", "string"}};
		return types_mapping;
	}

  public:
	using indexes_t = std::vector<std::string>;
	using aliases_t = indexes_t;
	using columns_t = std::vector<std::pair<std::string, std::pair<std::string, std::string>>>;
	using querybuilder_columns_t = std::vector<std::pair<std::string, std::string>>;
	using rows_t = std::vector<std::vector<std::string>>;
	using filter_paths_t = std::vector<std::string>;
	using visit_columns_f = std::function<void(const std::string& relative_name,
	                                           const std::string& absolute_name,
	                                           const std::string& type,
	                                           bool is_leaf,
	                                           bool is_last_child)>;

  public:
	PVElasticsearchAPI(const PVElasticsearchInfos& infos);
	~PVElasticsearchAPI();

  public:
	/** Check if the connection to the server is successful
	 *
	 * @param error Store any occured error if provided
	 *
	 * @return true if successfully connected to the server, false otherwise.
	 */
	bool check_connection(std::string* error = nullptr) const;

	/**
	 * Return elasticsearch version
	 */
	PVCore::PVVersion version() const;

	/** Fetch the list of indexes from the server
	 *
	 * @param error Store any occured error if provided
	 *
	 * @return the list of indexes
	 */
	indexes_t indexes(std::string* error = nullptr) const;

	/** Fetch the list of aliases from the server
	 *
	 * @param error Store any occured error if provided
	 *
	 * @return the list of aliases
	 */
	aliases_t aliases(std::string* error = nullptr) const;

	/** Fetch the list of columns for a given index provided in the Query object
	 *
	 * @param error Store any occured error if provided
	 *
	 * @return the list of columns
	 */
	querybuilder_columns_t querybuilder_columns(const std::string& filter_path = {},
	                                            std::string* error = nullptr) const;
	columns_t format_columns(const std::string& filter_path = {},
	                         std::string* error = nullptr) const;

	/** Narrow numeric types based on max value aggregation
	 *
	 * https://www.elastic.co/guide/en/elasticsearch/reference/master/search-aggregations-metrics-max-aggregation.html
	 */
	void narrow_numeric_types(columns_t& cols) const;

	/** Get the number of lines returned by a given query using the "count" API
	 *
	 * @param error Store any occured error if provided
	 *
	 * @return the query result count
	 */
	size_t count(const PVRush::PVElasticsearchQuery& query, std::string* error = nullptr) const;

	/*
	 * Return the number of shards for a given index
	 */
	size_t shards_count(const std::string& index, std::string* error = nullptr) const;

	/*
	 * Return the maximum batch size allowed by the server for scroll operations
	 */
	size_t max_result_window(const std::string& index) const;

	/*
	 * Returns the format associated to the selected columns of the mapping
	 */
	QDomDocument get_format_from_mapping() const;

  public:
	/** Get several batches of results from a query using Elasticsearch efficient scoll API.
	 *
	 * https://www.elastic.co/guide/en/elasticsearch/reference/current/search-request-scroll.html
	 *
	 * This method is used by the input plugin when exporting the result
	 * of a query, and by the source plugin when importing the result in
	 * INENDI Inspector.
	 *
	 * Note that the order of the lines is not the same as when imported
	 * due to a limitation of Elasticsearch.
	 *
	 * @param query the query to be executed
	 * @param rows_array a vector of batch results
	 * @param error Store any occured error if provided
	 *
	 * @return true if more data is available, false otherwise
	 */
	bool extract(const PVRush::PVElasticsearchQuery& query,
	             PVRush::PVElasticsearchAPI::rows_t& rows_array,
	             std::string* error = nullptr);

	/** Get the total count of expected results
	 *
	 * As this information is obtained from the parsing
	 * of a scroll initialization request, this method must therefore
	 * be called after the init_scroll method.
	 *
	 * @return  the total count of expected results
	 */
	size_t scroll_count() const;

	/**
	 * Release the scroll resources held by the server
	 *
	 * @return true if success, false otherwise
	 */
	bool clear_scroll();

  public:
	/** Convert json from QueryBuilder to json as ElasticSearch input
	 *
	 * @param rules : json from QueryBuilder
	 *
	 * @return json as Elasticsearch input
	 */
	std::string rules_to_json(const std::string& rules) const;

	/** Convert json from SQL editor to json as ElasticSearch input
	 *
	 * @param rules SQL query from SQL editor
	 *
	 * @return json as Elasticsearch input
	 */
	std::string sql_to_json(const std::string& sql, std::string* error = nullptr) const;

	/**
	 * @return true is Elasticsearch's SQL plugin is available on the server
	 */
	bool is_sql_available() const;

	/** Recursively visit all the columns of ES mapping (possibly filtered)
	 *
	 * @param f the function that is called against each column
	 * @param filter_path filtering to apply to the mapping
	 * @param error Store any occured error if provided
	 */
	void visit_columns(const visit_columns_f& f,
	                   const std::string& filter_path = {},
	                   std::string* error = nullptr) const;

  private:
	/** Execute the request previously prepared by the prepare_query method.
	 *
	 * This will store the returning scroll_id in the _scroll_id member variable
	 * in order to keep scrolling on the same query.
	 *
	 * @param query the query to be executed by the server
	 * @param error Store any occured error if provided
	 *
	 * @return true if the scroll was sucessfully initialized, false otherwise
	 */
	bool init_scroll(CURL* curl,
	                 const PVRush::PVElasticsearchQuery& query,
	                 const size_t slice_id,
	                 const size_t slice_count,
	                 const size_t max_result_window,
	                 std::string& json_buffer,
	                 std::string* error = nullptr);

	/**
	 * Keep the scroll_id contained in the received result batch (needed to query next result batch)
	 */
	void update_scroll_id(CURL* curl, const std::string& scroll_id) const;

	/** Get one batch of results from a query using Elasticsearch efficient scoll API.
	 *
	 * This method is by the pipeline of the extract method to allow a parallelized JSON parsing
	 *
	 * @param query the query to be executed by the server
	 * @param error Store any occured error if provided
	 */
	bool scroll(CURL* curl,
	            const PVRush::PVElasticsearchQuery& query,
	            bool init_scroll,
	            const size_t slice_id,
	            const size_t slice_count,
	            const size_t max_result_window,
	            std::string& json_data,
	            std::string* error = nullptr);

	/** Extract the raw data from a JSON scroll batch results.
	 *
	 * @param json_data the JSON content returned by the server
	 * @param rows a vector of strings representing the raw lines
	 */
	bool parse_scroll_results(CURL* curl,
	                          const std::string& json_data,
	                          std::string& scroll_id,
	                          rows_t& rows);

	void detect_time_formats(columns_t& cols) const;

  private:
	/** Setup cURL handler before executing a request.
	 *
	 * As Elasticsearch scroll API imply executing the exact same request over and over
	 * in order to exhaust the available results, there is no need to setup cURL over and
	 * over again, that's why the setup of the request is decorrelated from the request execution
	 *
	 * @param uri GET content
	 * @param body POST content
	 */
	void prepare_query(CURL* curl,
	                   const std::string& uri,
	                   const std::string& body = std::string()) const;

	/** Execute the request previously prepared by the prepare_query method.
	 *
	 * @param result the JSON content returned by the server
	 * @param error Store any occured error if provided
	 */
	bool perform_query(CURL* curl, std::string& result, std::string* error = nullptr) const;

  private:
	/**
	 * @return the server API endpoint needed to establish the communication with Elasticsearch
	 */
	std::string socket() const;

	/**
	 * Check if the returned JSON content contains errors
	 *
	 * @return true if error, false otherwise
	 */
	bool has_error(const rapidjson::Document& json, std::string* error = nullptr) const;

  private:
	CURL* _curl;                         // cURL request handler
	PVRush::PVElasticsearchInfos _infos; // Contains all the info to reach Elasticsearch server
	size_t _scroll_count = 0; // total count of event that will be returned by a scrolling session
	bool _init_scroll = true;
	size_t _slice_count = 0;
	size_t _max_result_window = 0;
	PVCore::PVVersion _version;
	mutable std::string _mapping_type; // for retro-compatibility
	using curlp_t = std::unique_ptr<CURL, std::function<void(CURL*)>>;
	std::vector<curlp_t> _curls;
	curl_slist* _curl_headers = nullptr;
	std::vector<std::string> _scroll_ids;
};
} // namespace PVRush

#endif // __PVELASTICSEARCHAPI_H__
