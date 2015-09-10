#ifndef __PVELASTICSEARCHJSONCONVERTER_H__
#define __PVELASTICSEARCHJSONCONVERTER_H__

#include <unordered_map>
#include <string>
#include <vector>

#include <rapidjson/document.h>                                                 
#include <rapidjson/writer.h>                                                   
#include <rapidjson/stringbuffer.h>

/** Converter object from QueryBuilder json to ElasticSearch json
 *
 * @note A class is required to keep internal state about in/not_in operators
 *
 * @note We perform a compilation from one DSL to another. It could be done
 * creating an AST and visiting it. As the conversion is pretty easy and 
 * simple (no optimisations or check) it is overkill...
 */
class PVElasticSearchJsonConverter
{
   public:
      /** Parse json input to be processed
       *
       * @param qb_rule : json input
       */
       PVElasticSearchJsonConverter(std::string const& qb_rule);

       /** Translate querybuilder json to elasticsearch json
        *
        * @return elasticsearch json input
        */
       std::string rules_to_json();

   private:
       rapidjson::Document _doc;  //!< json document with querybuilder json
       rapidjson::StringBuffer _strbuf;  //!< internal buffer to store elasticsearch json in construction document
       rapidjson::Writer<rapidjson::StringBuffer> _writer;  //!< internal object to create elasticsearch json file

       // We use a std::string as first parameter because const char* hash
       // use pointer as value.
       // const char* a = "ok", b = "ok" => std::hash(a) != std::hash(b)
       using map_t = std::unordered_map<std::string, std::vector<const char*>>;
       map_t _in_values;  //!< list of values with in operations for the current group
       map_t _not_in_values;  //!< list of values with "not in" operations for the current group

   private:
       /** Parse a node and add it's infomations in the json in progress.
        *
        * A node is a final object in the json tree.
        * It looks like:
        *
        *   {
        *       "id": "time_spent",
        *       "field": "time_spent",
        *       "type": "integer",
        *       "input": "text",
        *       "operator": "less",
        *       "value": "5"
        *   }
        *
        * @param obj : node to process
        */

       void parse_node(rapidjson::Value const& obj);
       /** Parse a condition and add it's infomations in the json in progress.
        *
        * A condition is an upper node in the json tree.
        * It looks like:
        *   {
        *       "condition": "AND",
        *       "rules": [ nodes_and_conditions]
        *   }
        *
        * @param obj : node to process
        *
        * @note in and not_in values are dumped once all rules are process.
        */
       void parse_condition(rapidjson::Value const& obj);

       /** Generate between json
        *
        * {"range": {"id": {"gte" : begin, "lte": end}}}
        */
       void between(rapidjson::Value const& id, rapidjson::Value const& begin,
                    rapidjson::Value const& end);

       /** Generic comparison for less/greater/less_or_equal/greater_or_equal */
       void compare(rapidjson::Value const& id, rapidjson::Value const& end, const char*);

       /** Generate ends_with json
        *
        * {"regexp": {"id": "*value*"}}
        */
       void contains(rapidjson::Value const& id, rapidjson::Value const& value);

       /** Generate is_empty json
        *
        * {"term": {"id": "value"}}
        */
       void equal(rapidjson::Value const& id, rapidjson::Value const& value);

       /** Generate greater json
        *
        * {"range": {"id": {"gte" : "value"}}}
        */
       void greater(rapidjson::Value const& id, rapidjson::Value const& end);

       /** Generate greater_or_equal json
        *
        * {"range": {"id": {"gt" : "value"}}}
        */
       void greater_or_equal(rapidjson::Value const& id, rapidjson::Value const& end);

       /** Push id/value in the in_values dict for later processing.
        */
       void in(rapidjson::Value const& id, rapidjson::Value const& values);

       /** Generate in json
        *
        * ForEach key, values in saved:
        * {"terms": {"key": [values...]}}
        */
       void inject_in(map_t const& saved);

       /** Generate is_empty json
        *
        * {"term": {"id": ""}}
        */
       void is_empty(rapidjson::Value const& id);

       /** Generate is_null json
        *
        * @todo : not implemented
        */
       void is_null(rapidjson::Value const& id);

       /** Generate less json
        *
        * {"range": {"id": {"lt" : "value"}}}
        */
       void less(rapidjson::Value const& id, rapidjson::Value const& end);

       /** Generate less_or_equal json
        *
        * {"range": {"id": {"lte" : "value"}}}
        */
       void less_or_equal(rapidjson::Value const& id, rapidjson::Value const& end);

       /** Push id/value in the not_in_values dict for later processing.
        */
       void not_in(rapidjson::Value const& id, rapidjson::Value const& values);

       /** Generate begins_with json
        *
        * {"prefix": {"id": "value"}}
        */
       void prefix(rapidjson::Value const& id, rapidjson::Value const& value);

       /** Generate ends_with json
        *
        * {"regexp": {"id": "value*"}}
        */
       void suffix(rapidjson::Value const& id, rapidjson::Value const& value);

       /** Generic method to negate a node
        *
        * {"not": contains_condition }
        */
       template <class F, class ...T>
       void not_(F fun, T && ... params);

       /** Check if a json object is a condition node.
        *
        * Condition contains "condition" keyword.
        */
       static bool is_condition(rapidjson::Value const& obj);

};


#endif
