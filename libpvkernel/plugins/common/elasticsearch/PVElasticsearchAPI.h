/**
 * @file PVElasticsearchAPI.h
 *
 * @copyright (C) Picviz Labs 2015
 */

#ifndef __PVELASTICSEARCHAPI_H__
#define __PVELASTICSEARCHAPI_H__

#include <vector>
#include <string>

#include "PVElasticsearchInfos.h"

#include <curl/curl.h>

namespace PVRush {

class PVElasticsearchQuery;

/** This class contains all the logic needed by Picviz Inspector
 *  to communicate with an Elasticsearch server.
 */
class PVElasticsearchAPI
{
public:
	static constexpr size_t DEFAULT_PORT = 9200;

public:
	using indexes_t = std::vector<std::string>;
	using columns_t = std::vector<std::pair<std::string, std::string>>;
	using rows_t = std::vector<std::string>;
	using rows_chunk_t = std::vector<rows_t>;

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

    /** Fetch the list of indexes from the server
     *
     * @param error Store any occured error if provided
     *
     * @return the list of indexes
     */
	indexes_t indexes(std::string* error = nullptr) const;

    /** Fetch the list of columns for a given index provided in the Query object
     *
     * @param error Store any occured error if provided
     *
     * @return the list of columns
     */
	columns_t columns(const PVRush::PVElasticsearchQuery& query, std::string* error = nullptr) const;

    /** Get the number of lines returned by a given query using the "count" API
     *
     * @param error Store any occured error if provided
     *
     * @return the query result count
     */
	size_t count(const PVRush::PVElasticsearchQuery& query, std::string* error = nullptr) const;

public:
	/** Get several batches of results from a query using Elasticsearch efficient scoll API.
	 *
	 * https://www.elastic.co/guide/en/elasticsearch/reference/current/search-request-scroll.html
	 *
	 * This method is used by the input plugin when exporting the result
	 * of a query, and by the source plugin when importing the result in
	 * Picviz Inspector.
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
	bool extract(
		const PVRush::PVElasticsearchQuery& query,
		PVRush::PVElasticsearchAPI::rows_chunk_t& rows_array,
		std::string* error = nullptr
	) const;

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
	bool init_scroll(const PVRush::PVElasticsearchQuery& query, std::string* error = nullptr);

	/** Get one batch of results from a query using Elasticsearch efficient scoll API.
	 *
	 * This method is by the pipeline of the extract method to allow a parallelized JSON parsing
	 *
	 * @param query the query to be executed by the server
	 * @param error Store any occured error if provided
	 */
	bool scroll(const PVRush::PVElasticsearchQuery& query, std::string& json_data, std::string* error = nullptr);

	/** Extract the raw data from a JSON scroll batch results.
	 *
	 * @param json_data the JSON content returned by the server
	 * @param rows a vector of strings representing the raw lines
	 */
	bool parse_scroll_results(const std::string& json_data, rows_t& rows) const;

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
	void prepare_query(const std::string& uri, const std::string& body = std::string()) const;

    /** Execute the request previously prepared by the prepare_query method.
     *
     * @param result the JSON content returned by the server
     * @param error Store any occured error if provided
     */
	bool perform_query(std::string& result, std::string* error = nullptr) const;

private:
	/**
	 * @return the server API endpoint needed to establish the communication with Elasticsearch
	 */
	std::string socket() const;

private:
	CURL* _curl; 							// cURL request handler
	PVRush::PVElasticsearchInfos _infos; 	// Contains all the info to reach Elasticsearch server
	std::string _scroll_id;					// ID returned by Elasticsearch scroll API
	size_t _scroll_count = 0;				// total count of event that will be returned by a scrolling session
};

}

#endif // __PVELASTICSEARCHAPI_H__
