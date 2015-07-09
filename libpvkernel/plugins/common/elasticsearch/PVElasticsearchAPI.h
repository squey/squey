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
	using indexes_t = std::vector<std::string>;
	using rows_t = std::vector<std::string>;

public:
	PVElasticsearchAPI(const PVElasticsearchQuery& query);
	~PVElasticsearchAPI();

public:
	bool scroll(std::string& json_data);
	bool clear_scroll();
	bool parse_results(const std::string& json_data, rows_t& rows) const;

public:
	indexes_t indexes() const;
	size_t count() const;

private:
	void prepare_query(const std::string& uri, const std::string& body = std::string()) const;
	bool perform_query(std::string& result) const;

private:
	bool init_scroll();

private:
	std::string socket() const;

private:
	CURL* _curl;
	const PVRush::PVElasticsearchQuery& _query;
	std::string _scroll_id;
};

}

#endif // __PVELASTICSEARCHAPI_H__
