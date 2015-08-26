#include "PVElasticsearchJsonConverter.h"

#include <functional>

///////////////////////////////////////////////////////////////////////////////
// PVElasticSearchJsonConverter constructor
///////////////////////////////////////////////////////////////////////////////

PVElasticSearchJsonConverter::PVElasticSearchJsonConverter(std::string const& qb_rule):
    _doc(), _strbuf(), _writer(_strbuf)
{
    _doc.Parse(qb_rule.c_str());
}

///////////////////////////////////////////////////////////////////////////////
// not_
///////////////////////////////////////////////////////////////////////////////

template <class F, class ...T>
void PVElasticSearchJsonConverter::not_(F fun, T && ... params)
{
   _writer.StartObject();
   _writer.Key("not");

   std::mem_fn(fun)(*this, std::forward<T>(params)...);

   _writer.EndObject();
}

///////////////////////////////////////////////////////////////////////////////
// in
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::in(rapidjson::Value const& id, rapidjson::Value const& value)
{
   _in_values[id.GetString()].push_back(value.GetString());
}

///////////////////////////////////////////////////////////////////////////////
// not_in
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::not_in(rapidjson::Value const& id, rapidjson::Value const& value)
{
   _not_in_values[id.GetString()].push_back(value.GetString());
}

///////////////////////////////////////////////////////////////////////////////
// suffix
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::suffix(rapidjson::Value const& id, rapidjson::Value const& value)
{
   _writer.StartObject();
   _writer.Key("regexp");
   _writer.StartObject();

   _writer.Key(id.GetString());
   std::string res = std::string("*") + value.GetString();
   _writer.String(res.c_str());

   _writer.EndObject();

   _writer.EndObject();
}

///////////////////////////////////////////////////////////////////////////////
// contains
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::contains(rapidjson::Value const& id, rapidjson::Value const& value)
{
   _writer.StartObject();

   _writer.Key("regexp");
   _writer.StartObject();

   _writer.Key(id.GetString());
   std::string res = std::string("*") + value.GetString() + "*";
   _writer.String(res.c_str());

   _writer.EndObject();

   _writer.EndObject();
}

///////////////////////////////////////////////////////////////////////////////
// prefix
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::prefix(rapidjson::Value const& id, rapidjson::Value const& value)
{
   _writer.StartObject();

   _writer.Key("prefix");
   _writer.StartObject();

   _writer.Key(id.GetString());
   _writer.String(value.GetString());

   _writer.EndObject();

   _writer.EndObject();
}

///////////////////////////////////////////////////////////////////////////////
// equal
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::equal(rapidjson::Value const& id, rapidjson::Value const& value)
{
   _writer.StartObject();

   _writer.Key("term");
   _writer.StartObject();

   _writer.Key(id.GetString());
   _writer.String(value.GetString());

   _writer.EndObject();

   _writer.EndObject();
}

///////////////////////////////////////////////////////////////////////////////
// between
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::between(rapidjson::Value const& id, rapidjson::Value const& begin, rapidjson::Value const& end)
{
   _writer.StartObject();

   _writer.Key("range");
   _writer.StartObject();

   _writer.Key(id.GetString());
   _writer.StartObject();

   _writer.Key("gte");
   _writer.Int(std::stol(begin.GetString()));
   _writer.Key("lte");
   _writer.Int(std::stol(end.GetString()));

   _writer.EndObject();

   _writer.EndObject();

   _writer.EndObject();
}

///////////////////////////////////////////////////////////////////////////////
// greater_or_equal
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::greater_or_equal(rapidjson::Value const& id, rapidjson::Value const& end)
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

void PVElasticSearchJsonConverter::less_or_equal(rapidjson::Value const& id, rapidjson::Value const& end)
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

void PVElasticSearchJsonConverter::compare(rapidjson::Value const& id, rapidjson::Value const& end, const char* op)
{
   _writer.StartObject();

   _writer.Key("range");
   _writer.StartObject();

   _writer.Key(id.GetString());
   _writer.StartObject();

   _writer.Key(op);
   _writer.Int(std::stol(end.GetString()));

   _writer.EndObject();

   _writer.EndObject();

   _writer.EndObject();
}

///////////////////////////////////////////////////////////////////////////////
// is_null
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::is_null(rapidjson::Value const& /* id */)
{
   assert(false && "Not Implemented");
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
   for(auto & k_v : saved) {
       _writer.StartObject();

       _writer.Key("terms");
       _writer.StartObject();

       _writer.Key(k_v.first.c_str());
       _writer.StartArray();

       for(const char* value: k_v.second) {
           _writer.String(value);
       }
       _writer.EndArray();

       _writer.EndObject();

       _writer.EndObject();
   }
}

///////////////////////////////////////////////////////////////////////////////
// is_condition
///////////////////////////////////////////////////////////////////////////////

bool PVElasticSearchJsonConverter::is_condition(rapidjson::Value const& obj)
{
   return obj.HasMember("condition");
}

///////////////////////////////////////////////////////////////////////////////
// parse_node
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::parse_node(rapidjson::Value const& obj)
{
   // Use a dispatching table to avoid too many comparison with strings
   enum TOKEN {EQUAL, NOT_EQUAL, BETWEEN, IN, NOT_IN, NOT_BEGINS_WITH,
      BEGINS_WITH, NOT_CONTAINS, CONTAINS, NOT_ENDS_WITH, ENDS_WITH, IS_NULL,
      IS_NOT_NULL, IS_EMPTY, IS_NOT_EMPTY, LESS, LESS_OR_EQUAL, GREATER,
      GREATER_OR_EQUAL};

   static const std::unordered_map<std::string, int> dispatch =
      {{"equal", EQUAL}, {"not_equal", NOT_EQUAL}, {"between", BETWEEN},
       {"in", IN}, {"not_in", NOT_IN}, {"not_begins_with", NOT_BEGINS_WITH},
       {"begins_with", BEGINS_WITH}, {"not_contains", NOT_CONTAINS},
       {"contains", CONTAINS}, {"not_ends_with", NOT_ENDS_WITH},
       {"ends_with", ENDS_WITH}, {"is_null", IS_NULL},
       {"is_not_null", IS_NOT_NULL}, {"is_empty", IS_EMPTY},
       {"is_not_empty", IS_NOT_EMPTY}, {"less", LESS},
       {"less_or_equal", LESS_OR_EQUAL}, {"greater", GREATER},
       {"greater_or_equal", GREATER_OR_EQUAL}};

   switch(dispatch.at(obj["operator"].GetString()))
   {
      case EQUAL:
         equal(obj["id"], obj["value"]);
         break;
      case NOT_EQUAL:
         not_(&PVElasticSearchJsonConverter::equal, obj["id"], obj["value"]);
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
         not_(&PVElasticSearchJsonConverter::prefix, obj["id"], obj["value"]);
         break;
      case BEGINS_WITH:
         prefix(obj["id"], obj["value"]);
         break;
      case NOT_CONTAINS:
         not_(&PVElasticSearchJsonConverter::contains, obj["id"], obj["value"]);
         break;
      case CONTAINS:
         contains(obj["id"], obj["value"]);
         break;
      case NOT_ENDS_WITH:
         not_(&PVElasticSearchJsonConverter::suffix, obj["id"], obj["value"]);
         break;
      case ENDS_WITH:
         suffix(obj["id"], obj["value"]);
         break;
      case IS_NULL:
         is_null(obj["id"]);
         break;
      case IS_NOT_NULL:
         not_(&PVElasticSearchJsonConverter::is_null, obj["id"]);
         break;
      case IS_EMPTY:
         is_empty(obj["id"]);
         break;
      case IS_NOT_EMPTY:
         not_(&PVElasticSearchJsonConverter::is_empty, obj["id"]);
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

///////////////////////////////////////////////////////////////////////////////
// parse_condition
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::parse_condition(rapidjson::Value const& obj)
{
   // Swap in and not_in values to keep it local for the current group
   map_t in_value_upper;
   map_t not_in_value_upper;

   std::swap(_in_values, in_value_upper);
   std::swap(_not_in_values, not_in_value_upper);

   _writer.StartObject();

   if(obj["condition"] == "AND") {
       _writer.Key("and");
   } else {
       assert(obj["condition"] == "OR" && "Uncorrect condition");
       _writer.Key("or");
   }

   _writer.StartArray();

   for (rapidjson::Value::ConstValueIterator itr = obj["rules"].Begin(); itr != obj["rules"].End(); ++itr) {
       if(is_condition(*itr)) {
           parse_condition(*itr);
       } else {
           parse_node(*itr);
       }
   }

   // Dump in and not_in values once they are fully gathered
   if(not _in_values.empty())
       inject_in(_in_values);
   if(not _not_in_values.empty())
       not_(&PVElasticSearchJsonConverter::inject_in, _not_in_values);

   _writer.EndArray();

   _writer.EndObject();

   // Get back previous in and not_in information
   std::swap(_in_values, in_value_upper);
   std::swap(_not_in_values, not_in_value_upper);
}

///////////////////////////////////////////////////////////////////////////////
// rules_to_json
///////////////////////////////////////////////////////////////////////////////

std::string PVElasticSearchJsonConverter::rules_to_json()
{
   if(not _doc.IsObject() or not is_condition(_doc))
      // If it is an empty input, output is empty too
      return "";

   _writer.StartObject();

   _writer.Key("query");
   _writer.StartObject();

   // Use filter as we don't want document score and it is cached in memory
   _writer.Key("filtered");
   _writer.StartObject();

   _writer.Key("filter");

   parse_condition(_doc);

   _writer.EndObject();

   _writer.EndObject();

   _writer.EndObject();

   return _strbuf.GetString();
}
