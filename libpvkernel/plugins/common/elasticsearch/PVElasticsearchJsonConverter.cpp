#include "PVElasticsearchJsonConverter.h"

///////////////////////////////////////////////////////////////////////////////
// PVElasticSearchJsonConverter constructor
///////////////////////////////////////////////////////////////////////////////

PVElasticSearchJsonConverter::PVElasticSearchJsonConverter(std::string const& qb_rule):
   PVQueryBuilderJsonConverter(qb_rule), _strbuf(), _writer(_strbuf)
{
}

///////////////////////////////////////////////////////////////////////////////
// pre_not_
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::pre_not_()
{
   _writer.StartObject();
   _writer.Key("not");
}

///////////////////////////////////////////////////////////////////////////////
// post_not_
///////////////////////////////////////////////////////////////////////////////

void PVElasticSearchJsonConverter::post_not_()
{
   _writer.EndObject();
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
