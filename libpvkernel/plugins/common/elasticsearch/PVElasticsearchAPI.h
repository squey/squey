/**
 * @file PVElasticsearchAPI.h
 *
 * @copyright (C) Picviz Labs 2015
 */

#ifndef __PVELASTICSEARCHAPI_H__
#define __PVELASTICSEARCHAPI_H__

#include <vector>
#include <string>

#include <curl/curl.h>

namespace PVRush {

class PVElasticsearchInfos;
class PVElasticsearchQuery;

class PVElasticsearchAPI
{
public:
	struct mapping_t {
		mapping_t(const std::string& n, const std::string& t) :name(n), type(t) {};
		std::string name;
		std::string type;
	}; // Types supported by the QueryBuilder are: [string, integer, double, date, time, datetime, boolean]

public:
	using indexes_t = std::vector<std::string>;
	using columns_t = std::vector<mapping_t>;
	using rows_t = std::vector<std::string>;

public:
	PVElasticsearchAPI(const PVElasticsearchInfos& infos);
	~PVElasticsearchAPI();

public:
	bool scroll(const PVRush::PVElasticsearchQuery& query, std::string& json_data);
	bool clear_scroll();
	bool parse_results(const std::string& json_data, rows_t& rows) const;
	std::string sql_to_json(const std::string& sql) const;

public:
	bool check_connection(std::string* error = nullptr) const;
	indexes_t indexes() const;
	columns_t columns(const PVRush::PVElasticsearchQuery& query) const;
	size_t count(const PVRush::PVElasticsearchQuery& query) const;
	bool is_sql_available() const;

private:
	void prepare_query(const std::string& uri, const std::string& body = std::string()) const;
	bool perform_query(std::string& result, std::string* error = nullptr) const;

private:
	bool init_scroll(const PVRush::PVElasticsearchQuery& query);

private:
	std::string socket() const;

private:
	CURL* _curl;
	const PVRush::PVElasticsearchInfos& _infos;
	std::string _scroll_id;
};

}

#endif // __PVELASTICSEARCHAPI_H__
