/**
 * \file
 *
 * Copyright (C) Picviz Labs 2015
 */

#ifndef __LIBPVKERNEL_PLUGINS_COMMON_SPLUNK_SPLUNKAPI_H__
#define __LIBPVKERNEL_PLUGINS_COMMON_SPLUNK_SPLUNKAPI_H__

#include "PVSplunkInfos.h"

#include <string>
#include <vector>
#include <mutex>

#include <curl/curl.h>
#include <unistd.h>

namespace PVRush
{

class PVSplunkQuery;

/**
 * This class contains all the logic needed by Picviz Inspector
 * to communicate with a Splunk server.
 */
class PVSplunkAPI
{
public:
	static constexpr size_t DEFAULT_PORT = 8089;

public:
	using columns_t = std::vector<std::pair<std::string, std::string>>;
	using rows_t = std::vector<std::string>;
	using strings_t = std::vector<std::string>;

public:
	PVSplunkAPI(const PVRush::PVSplunkInfos& infos);
	~PVSplunkAPI();

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
	strings_t indexes(std::string* error = nullptr) const;

    /** Fetch the list of hosts from the server
     *
     * @param error Store any occured error if provided
     *
     * @return the list of indexes
     */
	strings_t hosts(std::string* error = nullptr) const;

    /** Fetch the list of sourcetypes from the server
     *
     * @param error Store any occured error if provided
     *
     * @return the list of indexes
     */
	strings_t sourcetypes(std::string* error = nullptr) const;

    /** Fetch the list of columns (with their associated types)
     * from the server, filtered with indexes, hosts and sourcetypes
     *
     * @param error Store any occured error if provided
     *
     * @return the list of indexes
     */
	columns_t columns(std::string* error = nullptr) const;

    /** Get the number of lines returned by a given query
     *
     * @param error Store any occured error if provided
     *
     * @return the query result count
     */
	size_t count(const PVRush::PVSplunkQuery& query, std::string* error = nullptr) const;

	/** Get a batch of results from a query.
	 *
	 * This method is used by the input plugin when exporting the result
	 * of a query, and by the source plugin when importing the result in
	 * Picviz Inspector.
	 *
	 * Note that the order of the lines is not the same as when imported
	 * due to a limitation of Splunk.
	 *
	 * @param query the query to be executed
	 * @param data a row std::string containing a batch result
	 *
	 * @return true if more data is available, false otherwise
	 */
	bool extract(
		const PVRush::PVSplunkQuery& query,
		std::string& data,
		std::string* error = nullptr
	);

	/** Convert json from QueryBuilder to json as Splunk input
	 *
	 * @param rules : json from QueryBuilder
	 *
	 * @return json as Splunk input
	 */
	std::string rules_to_json(const std::string& rules) const;

public: // these functions are used by cURL callback functions
	void set_socket(int s);
	void set_timeout(long t);
	void append_data(const void* data, size_t size);
	size_t extract_buffer_size() const { return _data.size(); }

private:
    /** Configure cURL to perform an asynchronous extract request
     *
     * @param search_query the Splunk search string
     * @param error Store any occured error if provided
     */
	void prepare_extract(const std::string& search_query, std::string* error = nullptr);

    /** Execute the provided request in a synchronous way
     *
     * @param search_query the Splunk search string
     * @param result the content returned by the server
     * @param output_mode the wanted output fomat ("xml", "json", "raw")
     * @param error Store any occured error if provided
     */
	bool perform_query(
		const std::string& search_query,
		std::string& result,
		const std::string& output_mode= "json",
		std::string* error = nullptr
	) const;

private:
	 /** Return the Splunk export API URL to query
	 *
     * @param search_query the Splunk search string
     * @param output_mode the wanted output fomat ("xml", "json", "raw")
     * @param error Store any occured error if provided
     *
     * @return a string containing the Splunk export API URL to query
	 */
	std::string export_api_url(const std::string& search_query, const std::string& output_mode = "json", std::string* error =  nullptr) const;

	/**
	 * @return an empty search string filtered by the index, host and sourcetype
	 */
	std::string filtered_search() const;

private:
	/**
	 * Helper function used by indexes, hosts and sourcetypes methods
	 * to retrieve their content from the server.
	 *
	 * @param search_query the Splunk search string
	 * @param type the type of list ("host", "index" or "sourcetype")
	 * @param error Store any occured error if provided
	 *
	 * @return a list of the specified kind
	 */
	strings_t list(const std::string& search, const std::string& type, std::string* error =  nullptr) const;

private:
	/**
	 * Perform data polling on cURL export request
	 *
	 * @return true if more data is available, false otherwise
	 */
	bool poll();

	/**
	 * Retrieve a batch of data from an ongoing extract query
	 *
	 * In order to avoid blocking cURL from filling the internal buffer,
	 * we just std::move it and copy back the last line if not terminated
	 * by a new line character. That why the size of the data that must be
	 * consumed differs from the size of the returned buffer : because we
	 * must discard the last non-terminated line.
	 *
	 * @param buffer the batch of data
	 *
	 * @return the size of the data that must be consumed
	 */
	void extract_buffer(std::string& buffer);

private:
	PVRush::PVSplunkInfos 	_infos; // the splunk server related infos

	std::string    	_data; //! the internal buffer filled by an extract query
	fd_set         	_fdset; //! file descriptor needed by cURL polling API
	int           	_ongoing_extract_query = 0; //! used the cURL polling API
	bool           	_extract_canceled; //! used the to abort an angoing extract query
	std::mutex 		_mutex; //! used to protected the internel buffer when accessing it
	struct timeval 	_tv; //! used by cURL timeout
	CURLM*         	_multi; //! cURL multi request handler
	CURL*          	_easy; //! cURL easy request handler
	curl_socket_t  	_socket; //!cURL socket
};

} // namespace PVRush

#endif // __LIBPVKERNEL_PLUGINS_COMMON_SPLUNK_SPLUNKAPI_H__
