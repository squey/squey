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

#include "PVSplunkJsonConverter.h"

///////////////////////////////////////////////////////////////////////////////
// PVSplunkJsonConverter constructor
///////////////////////////////////////////////////////////////////////////////

PVSplunkJsonConverter::PVSplunkJsonConverter(std::string const& qb_rule)
    : PVQueryBuilderJsonConverter(qb_rule)
{
}

///////////////////////////////////////////////////////////////////////////////
// pre_not_
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::pre_not_()
{
	_writer << "NOT (";
}

///////////////////////////////////////////////////////////////////////////////
// post_not_
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::post_not_()
{
	_writer << ")";
}

///////////////////////////////////////////////////////////////////////////////
// suffix
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::suffix(rapidjson::Value const& id, rapidjson::Value const& value)
{
	_writer << id.GetString() << "=\"*" << value.GetString() << "\"";
}

///////////////////////////////////////////////////////////////////////////////
// contains
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::contains(rapidjson::Value const& id, rapidjson::Value const& value)
{
	_writer << id.GetString() << "=\"*" << value.GetString() << "*\"";
}

///////////////////////////////////////////////////////////////////////////////
// prefix
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::prefix(rapidjson::Value const& id, rapidjson::Value const& value)
{
	_writer << id.GetString() << "=\"" << value.GetString() << "*\"";
}

///////////////////////////////////////////////////////////////////////////////
// equal
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::equal(rapidjson::Value const& id, rapidjson::Value const& value)
{
	_writer << id.GetString() << "=\"" << value.GetString() << "\"";
}

///////////////////////////////////////////////////////////////////////////////
// between
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::between(rapidjson::Value const& id,
                                    rapidjson::Value const& begin,
                                    rapidjson::Value const& end)
{
	_writer << "(" << id.GetString() << ">=" << begin.GetString() << "AND " << id.GetString()
	        << "<=" << end.GetString() << ")";
}

///////////////////////////////////////////////////////////////////////////////
// greater_or_equal
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::greater_or_equal(rapidjson::Value const& id,
                                             rapidjson::Value const& begin)
{
	_writer << id.GetString() << ">=" << begin.GetString();
}

///////////////////////////////////////////////////////////////////////////////
// greater
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::greater(rapidjson::Value const& id, rapidjson::Value const& begin)
{
	_writer << id.GetString() << ">" << begin.GetString();
}

///////////////////////////////////////////////////////////////////////////////
// less_or_equal
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::less_or_equal(rapidjson::Value const& id, rapidjson::Value const& end)
{
	_writer << id.GetString() << "<=" << end.GetString();
}

///////////////////////////////////////////////////////////////////////////////
// less
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::less(rapidjson::Value const& id, rapidjson::Value const& end)
{
	_writer << id.GetString() << "<" << end.GetString();
}

///////////////////////////////////////////////////////////////////////////////
// inject_in
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::inject_in(map_t const& saved)
{
	bool first = true;
	_writer << "(";
	for (auto& k_v : saved) {
		for (const char* value : k_v.second) {
			if (first) {
				first = false;
			} else {
				_writer << " OR ";
			}
			_writer << k_v.first << "=" << value;
		}
	}
	_writer << ")";
}

///////////////////////////////////////////////////////////////////////////////
// parse_condition
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::parse_condition(rapidjson::Value const& obj)
{
	// Swap in and not_in values to keep it local for the current group
	map_t in_value_upper;
	map_t not_in_value_upper;

	std::swap(_in_values, in_value_upper);
	std::swap(_not_in_values, not_in_value_upper);

	_writer << "(";

	std::string cond = "AND";

	if (obj["condition"] == "OR") {
		cond = "OR";
	} else {
		assert(obj["condition"] == "AND" && "Uncorrect condition");
	}

	bool first = true;
	for (rapidjson::Value::ConstValueIterator itr = obj["rules"].Begin(); itr != obj["rules"].End();
	     ++itr) {
		if (first) {
			first = false;
		} else {
			_writer << cond;
		}
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
		not_<PVSplunkJsonConverter>(&PVSplunkJsonConverter::inject_in, _not_in_values);

	_writer << ")";

	// Get back previous in and not_in information
	std::swap(_in_values, in_value_upper);
	std::swap(_not_in_values, not_in_value_upper);
}

///////////////////////////////////////////////////////////////////////////////
// rules_to_json
///////////////////////////////////////////////////////////////////////////////

std::string PVSplunkJsonConverter::rules_to_json()
{
	if (not _doc.IsObject() or not is_condition(_doc))
		// If it is an empty input, output is empty too
		return "";

	parse_condition(_doc);

	return _writer.str();
}
