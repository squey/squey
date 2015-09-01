#include "PVSplunkJsonConverter.h"

///////////////////////////////////////////////////////////////////////////////
// PVSplunkJsonConverter constructor
///////////////////////////////////////////////////////////////////////////////

PVSplunkJsonConverter::PVSplunkJsonConverter(std::string const& qb_rule):
    PVQueryBuilderJsonConverter(qb_rule)
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

void PVSplunkJsonConverter::between(rapidjson::Value const& id, rapidjson::Value const& begin, rapidjson::Value const& end)
{
   _writer << "(" << id.GetString() << ">=" << begin.GetString() << "AND "
                  << id.GetString() << "<=" << end.GetString() << ")";
}

///////////////////////////////////////////////////////////////////////////////
// greater_or_equal
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::greater_or_equal(rapidjson::Value const& id, rapidjson::Value const& begin)
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
// is_null
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::is_null(rapidjson::Value const& /* id */)
{
   assert(false && "Not Implemented");
}

///////////////////////////////////////////////////////////////////////////////
// inject_in
///////////////////////////////////////////////////////////////////////////////

void PVSplunkJsonConverter::inject_in(map_t const& saved)
{
   bool first = true;
   _writer << "(";
   for(auto & k_v : saved) {
       for(const char* value: k_v.second) {
          if(first) {
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

   if(obj["condition"] == "OR") {
      cond = "OR";
   } else {
      assert(obj["condition"] == "AND" && "Uncorrect condition");
   }

   bool first = true;
   for (rapidjson::Value::ConstValueIterator itr = obj["rules"].Begin(); itr != obj["rules"].End(); ++itr) {
      if(first) {
         first = false;
      } else {
         _writer << cond;
      }
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
       not_(&PVSplunkJsonConverter::inject_in, _not_in_values);

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
   if(not _doc.IsObject() or not is_condition(_doc))
      // If it is an empty input, output is empty too
      return "";

   parse_condition(_doc);

   return _writer.str();
}
