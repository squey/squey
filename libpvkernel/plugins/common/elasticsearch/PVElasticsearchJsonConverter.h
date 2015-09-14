#ifndef __PVELASTICSEARCHJSONCONVERTER_H__
#define __PVELASTICSEARCHJSONCONVERTER_H__

#include <string>

#include <rapidjson/writer.h>                                                   
#include <rapidjson/stringbuffer.h>

#include <pvkernel/core/PVQueryBuilderJsonConverter.h>

/** Converter object from QueryBuilder json to ElasticSearch json
 */
class PVElasticSearchJsonConverter: public PVCore::PVQueryBuilderJsonConverter
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
       rapidjson::StringBuffer _strbuf;  //!< internal buffer to store elasticsearch json in construction document
       rapidjson::Writer<rapidjson::StringBuffer> _writer;  //!< internal object to create elasticsearch json file

   private:
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
       void pre_not_();
       void post_not_();

};


#endif
