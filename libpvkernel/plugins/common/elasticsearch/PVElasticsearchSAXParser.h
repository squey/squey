/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2017
 */

#include <unordered_map>
#include <vector>
#include <string>

#include <rapidjson/reader.h>

#include <boost/algorithm/string/join.hpp>

#ifndef __PVELASTICSEARCHSAXPARSER_H__
#define __PVELASTICSEARCHSAXPARSER_H__

/*
 * JSON SAX parser class used to parse Elasticsearch scroll batch content
 *
 * The content structure is as follow :
 * ```
 * {
 *   "hits":
 *   {
 *     "hits": [
 *     {
 *       "_source":
 *       {
 *         "leaf_key1": "value1",
 *         "leaf_key2": "value2",
 *         "node_key1":
 *         {
 *           "leaf_key3": "value3",
 *           "leaf_key4": "value4",
 *           "node_key2":
 *           {
 *           	"leaf_key5": "value5"
 *           	...
 *           },
 *           ...
 *         },
 *         "leaf_key6": "value6,
 *         ...
 *       },
 *       "_source":
 *       {
 *          ...
 *       },
 *       ...
 *     ]
 *   }
 * }
 * ```
 */

class PVElasticsearchSAXParser : public rapidjson::BaseReaderHandler<>
{
  public:
	PVElasticsearchSAXParser(const std::string& filter_path,
	                         PVRush::PVElasticsearchAPI::rows_t& rows)
	    : _state(), _rows(rows), _col_idx(0), _pop_name(false)
	{
		size_t idx = 0;
		std::vector<std::string> columns;
		boost::algorithm::split(columns, filter_path, boost::is_any_of(","));
		for (const std::string& col : columns) {
			_order.emplace(col, idx++);
		}
	}

  public:
	PVRush::PVElasticsearchAPI::rows_t& get_rows() { return _rows; }
	const std::string& scroll_id() const { return _scroll_id; }
	uint32_t total() const { return _total; }

  public:
	/*
	 * Method called each time an object starts
	 * note : called *after* we get the key name
	 */
	bool StartObject()
	{
		_pop_name = false;

		if (_state == ExpectObjectSourceStart) {
			_state = ExpectColumnName;
		}

		return true;
	}

	/*
	 * Method called each time an object ends
	 */
	bool EndObject(size_t)
	{
		if (_state == ExpectColumnName) {
			_column_name.pop_back();
			if (_column_name.size() == 0) {
				_state = ArrayHitsStarted;
			}
		}

		return true;
	}

	/*
	 * Method called each time a key is encountered
	 * note : at this point we don't know yet if the key is representing a node or a leaf
	 */
	bool Key(const char* str, size_t len, bool /*copy*/)
	{
		if (_state == Start and strncmp(str, "_scroll_id", len) == 0) {
			_state = ExpectScrollId;
		} else if (_state == ExpectObjectHitsStart and strncmp(str, "hits", len) == 0) {
			_state = ObjectHitsStarted;
		} else if (_state == ObjectHitsStarted and strncmp(str, "total", len) == 0) {
			_state = ExpectTotalValue;
		} else if (_state == ExpectArrayHitsStart and strncmp(str, "hits", len) == 0) {
			_state = ArrayHitsStarted;
		} else if (_state == ArrayHitsStarted and strncmp(str, "_source", len) == 0) {
			_state = ExpectObjectSourceStart;
			_rows.emplace_back();
			_rows.back().resize(_order.size());
		} else if (_state == ExpectColumnName) {

			// update absolute name
			if (_pop_name) {
				_column_name.pop_back();
			}
			_column_name.emplace_back(str, len);
			_pop_name = true;

			// retrieve field order according to mapping
			auto it = _order.find(get_name());
			if (it != _order.end()) { // note : lookup fails if we are not on a leaf
				_col_idx = it->second;
			}
		}

		return true;
	}

  public:
	bool String(const char* str, size_t len, bool)
	{
		if (_state == ExpectScrollId) {
			_scroll_id = str;
			_state = ExpectObjectHitsStart;
		} else if (_state == ExpectColumnName) {
			_rows.back()[_col_idx] = std::string(str, len);
		}
		return true;
	}

	bool Null()
	{
		if (_state == ExpectColumnName) {
			_rows.back()[_col_idx] = "";
		}
		return true;
	}

	bool Bool(bool value)
	{
		if (_state == ExpectColumnName) {
			_rows.back()[_col_idx] = value ? "true" : "false";
		}
		return true;
	}

	bool Int(int value)
	{
		if (_state == ExpectColumnName) {
			_rows.back()[_col_idx] = std::to_string(value);
		}
		return true;
	}

	bool Uint(uint32_t value)
	{
		if (_state == ExpectTotalValue) {
			_total = value;
			_state = ExpectArrayHitsStart;
		}
		if (_state == ExpectColumnName) {
			_rows.back()[_col_idx] = std::to_string(value);
		}
		return true;
	}

	bool Int64(int64_t value)
	{
		if (_state == ExpectColumnName) {
			_rows.back()[_col_idx] = std::to_string(value);
		}
		return true;
	}

	bool Uint64(uint64_t value)
	{
		if (_state == ExpectColumnName) {
			_rows.back()[_col_idx] = std::to_string(value);
		}
		return true;
	}

  private:
	/*
	 * Return the absolute name of the current column (ex : parent_name.parent_name.column_name)
	 */
	std::string get_name() const { return boost::algorithm::join(_column_name, "."); }

  private:
	enum EState {
		Start,
		ExpectScrollId,
		ExpectObjectHitsStart,
		ObjectHitsStarted,
		ExpectTotalValue,
		ExpectArrayHitsStart,
		ArrayHitsStarted,
		ExpectObjectSourceStart,
		ObjectSourceStarted,
		ExpectColumnName
	} _state;

	PVRush::PVElasticsearchAPI::rows_t& _rows;
	std::string _scroll_id;
	uint32_t _total = 0;

	// used to reorder fields according to mapping order
	std::unordered_map<std::string, size_t>
	    _order; // mapping between absolute column names and their indices
	std::vector<std::string> _column_name; // stack of current column ancestor names
	size_t _col_idx;                       // column index used to store column value into `_rows`
	bool _pop_name; // whenever should discard last column name from the stack
};

#endif // __PVELASTICSEARCHSAXPARSER_H__
