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

class PVElasticsearchAPI
{
public:
	// Types supported by the QueryBuilder are: [string, integer, double, date, time, datetime, boolean]
	// Types supported by Elasticsearch are: [string, integer/long, float/double, boolean, and null]

public:
	using indexes_t = std::vector<std::string>;
	using columns_t = std::vector<std::pair<std::string, std::string>>;
	using rows_t = std::vector<std::string>;
	using rows_chunk_t = std::vector<rows_t>;

public:
	PVElasticsearchAPI(const PVElasticsearchInfos& infos);
	~PVElasticsearchAPI();

public:
	bool scroll(const PVRush::PVElasticsearchQuery& query, std::string& json_data, std::string* error = nullptr);
	size_t scroll_count() const;
	bool clear_scroll();
	bool parse_scroll_results(const std::string& json_data, rows_t& rows) const;
	std::string sql_to_json(const std::string& sql, std::string* error = nullptr) const;

	/*
	 * Convert json from QueryBuilder to json as ElasticSearch input
	 *
	 * @param rules : json from QueryBuilder
	 *
	 * @return json as Elasticsearch input
	 */
	std::string rules_to_json(const std::string& rules) const;

public:
	bool check_connection(std::string* error = nullptr) const;
	indexes_t indexes(std::string* error = nullptr) const;
	columns_t columns(const PVRush::PVElasticsearchQuery& query, std::string* error = nullptr) const;
	size_t count(const PVRush::PVElasticsearchQuery& query, std::string* error = nullptr) const;
	bool is_sql_available() const;
	bool extract(
		const PVRush::PVElasticsearchQuery& query,
		PVRush::PVElasticsearchAPI::rows_chunk_t& rows_array,
		std::string* error = nullptr
	) const;

private:
	void prepare_query(const std::string& uri, const std::string& body = std::string()) const;
	bool perform_query(std::string& result, std::string* error = nullptr) const;

private:
	bool init_scroll(const PVRush::PVElasticsearchQuery& query, std::string* error = nullptr);

private:
	std::string socket() const;

private:
	CURL* _curl;
	PVRush::PVElasticsearchInfos _infos;
	std::string _scroll_id;
	size_t _scroll_count = 0;
};

}

#endif // __PVELASTICSEARCHAPI_H__
