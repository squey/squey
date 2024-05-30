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

#include "pvkernel/core/PVQueryBuilderJsonConverter.h"

namespace PVCore
{

///////////////////////////////////////////////////////////////////////////////
// PVQueryBuilderJsonConverter constructor
///////////////////////////////////////////////////////////////////////////////

PVQueryBuilderJsonConverter::PVQueryBuilderJsonConverter(std::string const& qb_rule) : _doc()
{
	_doc.Parse(qb_rule.c_str());
}

///////////////////////////////////////////////////////////////////////////////
// in
///////////////////////////////////////////////////////////////////////////////

void PVQueryBuilderJsonConverter::in(rapidjson::Value const& id, rapidjson::Value const& value)
{
	_in_values[id.GetString()].push_back(value.GetString());
}

///////////////////////////////////////////////////////////////////////////////
// not_in
///////////////////////////////////////////////////////////////////////////////

void PVQueryBuilderJsonConverter::not_in(rapidjson::Value const& id, rapidjson::Value const& value)
{
	_not_in_values[id.GetString()].push_back(value.GetString());
}

///////////////////////////////////////////////////////////////////////////////
// is_condition
///////////////////////////////////////////////////////////////////////////////

bool PVQueryBuilderJsonConverter::is_condition(rapidjson::Value const& obj)
{
	return obj.HasMember("condition");
}

///////////////////////////////////////////////////////////////////////////////
// parse_node
///////////////////////////////////////////////////////////////////////////////

void PVQueryBuilderJsonConverter::parse_node(rapidjson::Value const& obj)
{
	// Use a dispatching table to avoid too many comparison with strings
	enum TOKEN {
		EQUAL,
		NOT_EQUAL,
		BETWEEN,
		IN,
		NOT_IN,
		NOT_BEGINS_WITH,
		BEGINS_WITH,
		NOT_CONTAINS,
		CONTAINS,
		NOT_ENDS_WITH,
		ENDS_WITH,
		IS_NULL,
		IS_NOT_NULL,
		IS_EMPTY,
		IS_NOT_EMPTY,
		LESS,
		LESS_OR_EQUAL,
		GREATER,
		GREATER_OR_EQUAL
	};

	static const std::unordered_map<std::string, int> dispatch = {
	    {"equal", EQUAL},
	    {"not_equal", NOT_EQUAL},
	    {"between", BETWEEN},
	    {"in", IN},
	    {"not_in", NOT_IN},
	    {"not_begins_with", NOT_BEGINS_WITH},
	    {"begins_with", BEGINS_WITH},
	    {"not_contains", NOT_CONTAINS},
	    {"contains", CONTAINS},
	    {"not_ends_with", NOT_ENDS_WITH},
	    {"ends_with", ENDS_WITH},
	    {"is_empty", IS_EMPTY},
	    {"is_not_empty", IS_NOT_EMPTY},
	    {"less", LESS},
	    {"less_or_equal", LESS_OR_EQUAL},
	    {"greater", GREATER},
	    {"greater_or_equal", GREATER_OR_EQUAL}};

	switch (dispatch.at(obj["operator"].GetString())) {
	case EQUAL:
		equal(obj["id"], obj["value"]);
		break;
	case NOT_EQUAL:
		not_<PVQueryBuilderJsonConverter>(&PVQueryBuilderJsonConverter::equal, obj["id"],
		                                  obj["value"]);
		break;
	case BETWEEN:
		between(obj["id"], obj["value"][0], obj["value"][1]);
		break;
	case IN:
		in(obj["id"], obj["value"]);
		break;
	case NOT_IN:
		not_in(obj["id"], obj["value"]);
		break;
	case NOT_BEGINS_WITH:
		not_<PVQueryBuilderJsonConverter>(&PVQueryBuilderJsonConverter::prefix, obj["id"],
		                                  obj["value"]);
		break;
	case BEGINS_WITH:
		prefix(obj["id"], obj["value"]);
		break;
	case NOT_CONTAINS:
		not_<PVQueryBuilderJsonConverter>(&PVQueryBuilderJsonConverter::contains, obj["id"],
		                                  obj["value"]);
		break;
	case CONTAINS:
		contains(obj["id"], obj["value"]);
		break;
	case NOT_ENDS_WITH:
		not_<PVQueryBuilderJsonConverter>(&PVQueryBuilderJsonConverter::suffix, obj["id"],
		                                  obj["value"]);
		break;
	case ENDS_WITH:
		suffix(obj["id"], obj["value"]);
		break;
	case IS_EMPTY:
		is_empty(obj["id"]);
		break;
	case IS_NOT_EMPTY:
		not_<PVQueryBuilderJsonConverter>(&PVQueryBuilderJsonConverter::is_empty, obj["id"]);
		break;
	case LESS:
		less(obj["id"], obj["value"]);
		break;
	case LESS_OR_EQUAL:
		less_or_equal(obj["id"], obj["value"]);
		break;
	case GREATER:
		greater(obj["id"], obj["value"]);
		break;
	case GREATER_OR_EQUAL:
		greater_or_equal(obj["id"], obj["value"]);
		break;
	}
}
} // namespace PVCore
