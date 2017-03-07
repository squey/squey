/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include "PVElasticsearchJsonConverter.h"

///////////////////////////////////////////////////////////////////////////////
// PVElasticSearchJsonConverter constructor
///////////////////////////////////////////////////////////////////////////////

PVElasticSearchJsonConverter::PVElasticSearchJsonConverter(const PVCore::PVVersion& version,
                                                           std::string const& qb_rule)
    : PVQueryBuilderJsonConverter(qb_rule)
    , _strbuf()
    , _writer(_strbuf)
    , _writer_negate(_strbuf_not)
    , _current_writer(&_writer)
    , _version(version)
{
}

///////////////////////////////////////////////////////////////////////////////
// pre_not_
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::pre_not_()
{
	if (_version < PVCore::PVVersion(5, 0, 0)) {
		_current_writer->StartObject();
		_current_writer->Key("not");
	} else {
		_current_writer = &_writer_negate;
	}
}

///////////////////////////////////////////////////////////////////////////////
// post_not_
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::post_not_()
{
	if (_version < PVCore::PVVersion(5, 0, 0)) {
		_current_writer->EndObject();
	} else {
		_negated_terms.emplace_back(_strbuf_not.GetString());
		_strbuf_not.Clear();
		_writer_negate.Reset(_strbuf_not);
		_current_writer = &_writer;
	}
}

///////////////////////////////////////////////////////////////////////////////
// suffix
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::suffix(rapidjson::Value const& id, rapidjson::Value const& value)
{
	_current_writer->StartObject();
	_current_writer->Key("regexp");
	_current_writer->StartObject();

	_current_writer->Key(id.GetString());
	std::string res = std::string("*") + value.GetString();
	_current_writer->String(res.c_str());

	_current_writer->EndObject();

	_current_writer->EndObject();
}

///////////////////////////////////////////////////////////////////////////////
// contains
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::contains(rapidjson::Value const& id,
                                            rapidjson::Value const& value)
{
	_current_writer->StartObject();

	_current_writer->Key("regexp");
	_current_writer->StartObject();

	_current_writer->Key(id.GetString());
	std::string res = std::string("*") + value.GetString() + "*";
	_current_writer->String(res.c_str());

	_current_writer->EndObject();

	_current_writer->EndObject();
}

///////////////////////////////////////////////////////////////////////////////
// prefix
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::prefix(rapidjson::Value const& id, rapidjson::Value const& value)
{
	_current_writer->StartObject();

	_current_writer->Key("prefix");
	_current_writer->StartObject();

	_current_writer->Key(id.GetString());
	_current_writer->String(value.GetString());

	_current_writer->EndObject();

	_current_writer->EndObject();
}

///////////////////////////////////////////////////////////////////////////////
// equal
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::equal(rapidjson::Value const& id, rapidjson::Value const& value)
{
	_current_writer->StartObject();

	_current_writer->Key("term");
	_current_writer->StartObject();

	_current_writer->Key(id.GetString());
	_current_writer->String(value.GetString());

	_current_writer->EndObject();

	_current_writer->EndObject();
}

///////////////////////////////////////////////////////////////////////////////
// between
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::between(rapidjson::Value const& id,
                                           rapidjson::Value const& begin,
                                           rapidjson::Value const& end)
{
	_current_writer->StartObject();

	_current_writer->Key("range");
	_current_writer->StartObject();

	_current_writer->Key(id.GetString());
	_current_writer->StartObject();

	_current_writer->Key("gte");
	_current_writer->Int(std::stol(begin.GetString()));
	_current_writer->Key("lte");
	_current_writer->Int(std::stol(end.GetString()));

	_current_writer->EndObject();

	_current_writer->EndObject();

	_current_writer->EndObject();
}

///////////////////////////////////////////////////////////////////////////////
// greater_or_equal
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::greater_or_equal(rapidjson::Value const& id,
                                                    rapidjson::Value const& end)
{
	compare(id, end, "gte");
}

///////////////////////////////////////////////////////////////////////////////
// greater
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::greater(rapidjson::Value const& id, rapidjson::Value const& end)
{
	compare(id, end, "gt");
}

///////////////////////////////////////////////////////////////////////////////
// less_or_equal
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::less_or_equal(rapidjson::Value const& id,
                                                 rapidjson::Value const& end)
{
	compare(id, end, "lte");
}

///////////////////////////////////////////////////////////////////////////////
// less
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::less(rapidjson::Value const& id, rapidjson::Value const& end)
{
	compare(id, end, "lt");
}

///////////////////////////////////////////////////////////////////////////////
// compare
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::compare(rapidjson::Value const& id,
                                           rapidjson::Value const& end,
                                           const char* op)
{
	_current_writer->StartObject();

	_current_writer->Key("range");
	_current_writer->StartObject();

	_current_writer->Key(id.GetString());
	_current_writer->StartObject();

	_current_writer->Key(op);
	_current_writer->Int(std::stol(end.GetString()));

	_current_writer->EndObject();

	_current_writer->EndObject();

	_current_writer->EndObject();
}

///////////////////////////////////////////////////////////////////////////////
// is_empty
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::is_empty(rapidjson::Value const& id)
{
	equal(id, rapidjson::Value(""));
}

///////////////////////////////////////////////////////////////////////////////
// inject_in
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::inject_in(map_t const& saved)
{
	for (auto& k_v : saved) {
		_current_writer->StartObject();

		_current_writer->Key("terms");
		_current_writer->StartObject();

		_current_writer->Key(k_v.first.c_str());
		_current_writer->StartArray();

		for (const char* value : k_v.second) {
			_current_writer->String(value);
		}
		_current_writer->EndArray();

		_current_writer->EndObject();

		_current_writer->EndObject();
	}
}

///////////////////////////////////////////////////////////////////////////////
// inject_not
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::inject_not(std::vector<std::string> const& saved)
{
	for (auto& k_v : saved) {
		_current_writer->StartObject();

		_current_writer->Key("must_not");
		_current_writer->StartArray();

		for (const std::string& json_buffer : saved) {
			_current_writer->String(json_buffer.c_str());
		}
		_current_writer->EndArray();

		_current_writer->EndObject();
	}
}

///////////////////////////////////////////////////////////////////////////////
// parse_condition
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::parse_condition(rapidjson::Value const& obj)
{
	// Swap in and not_in values to keep it local for the current group
	map_t in_value_upper;
	map_t not_in_value_upper;
	std::vector<std::string> negated_terms;

	std::swap(_in_values, in_value_upper);
	std::swap(_not_in_values, not_in_value_upper);
	std::swap(negated_terms, _negated_terms);

	if (_version >= PVCore::PVVersion(5, 0, 0)) {
		_current_writer->StartObject();
		_current_writer->Key("bool");
	}

	_current_writer->StartObject();

	if (obj["condition"] == "AND") {
		if (_version < PVCore::PVVersion(5, 0, 0)) {
			_current_writer->Key("and");
		} else {
			_current_writer->Key("must");
		}
	} else {
		assert(obj["condition"] == "OR" && "Uncorrect condition");
		if (_version < PVCore::PVVersion(5, 0, 0)) {
			_current_writer->Key("or");
		} else {
			_current_writer->Key("should");
		}
	}

	_current_writer->StartArray();

	for (rapidjson::Value::ConstValueIterator itr = obj["rules"].Begin(); itr != obj["rules"].End();
	     ++itr) {
		if (is_condition(*itr)) {
			parse_condition(*itr);
		} else {
			parse_node(*itr);
		}
	}

	// Dump in and not_in values once they are fully gathered
	if (not _in_values.empty())
		inject_in(_in_values);
	if (not _not_in_values.empty())
		not_<PVElasticSearchJsonConverter>(&PVElasticSearchJsonConverter::inject_in,
		                                   _not_in_values);
	if (_version >= PVCore::PVVersion(5, 0, 0) and not _negated_terms.empty()) {
		inject_not(negated_terms);
	}

	_current_writer->EndArray();

	_current_writer->EndObject();

	if (_version >= PVCore::PVVersion(5, 0, 0)) {
		_current_writer->EndObject();
	}

	// Get back previous in and not_in information
	std::swap(_in_values, in_value_upper);
	std::swap(_not_in_values, not_in_value_upper);
	std::swap(negated_terms, _negated_terms);
}

///////////////////////////////////////////////////////////////////////////////
// rules_to_json
///////////////////////////////////////////////////////////////////////////////

std::string PVElasticSearchJsonConverter::rules_to_json()
{
	if (not _doc.IsObject() or not is_condition(_doc))
		// If it is an empty input, output is empty too
		return "";

	_current_writer->StartObject();

	_current_writer->Key("query");
	_current_writer->StartObject();

	// Use filter as we don't want document score and it is cached in memory
	if (_version >= PVCore::PVVersion(5, 0, 0)) {
		_current_writer->Key("constant_score");
	} else {
		_current_writer->Key("filtered");
	}
	_current_writer->StartObject();

	_current_writer->Key("filter");

	parse_condition(_doc);

	_current_writer->EndObject();

	_current_writer->EndObject();

	_current_writer->EndObject();

	return _strbuf.GetString();
}
