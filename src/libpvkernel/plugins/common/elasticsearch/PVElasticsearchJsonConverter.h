/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVELASTICSEARCHJSONCONVERTER_H__
#define __PVELASTICSEARCHJSONCONVERTER_H__

#include <string>

#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include <pvkernel/core/PVQueryBuilderJsonConverter.h>
#include <pvkernel/core/PVVersion.h>

/** Converter object from QueryBuilder json to ElasticSearch json
 */
class PVElasticSearchJsonConverter : public PVCore::PVQueryBuilderJsonConverter
{
  private:
	class PVRawWriter : public rapidjson::Writer<rapidjson::StringBuffer>
	{
	  public:
		using rapidjson::Writer<rapidjson::StringBuffer>::Writer;

		bool RawString(const Ch* str)
		{
			for (size_t i = 0; str[i] != '\0'; i++) {
				os_->Put(str[i]);
			}
			return true;
		}
	};

  public:
	/** Parse json input to be processed
	 *
	 * @param qb_rule : json input
	 */
	PVElasticSearchJsonConverter(const PVCore::PVVersion& version, std::string const& qb_rule);

	/** Translate querybuilder json to elasticsearch json
	 *
	 * @return elasticsearch json input
	 */
	std::string rules_to_json() override;

  private:
	using not_map_t = std::unordered_map<std::string, const char*>;

	rapidjson::StringBuffer
	    _strbuf; //!< internal buffer to store elasticsearch json in construction document
	rapidjson::StringBuffer
	    _strbuf_not;     //!< internal buffer to store elasticsearch json in construction document
	PVRawWriter _writer; // writer for normal operations
	PVRawWriter _writer_negate; // writer for negated operations

	PVRawWriter* _current_writer; // pointer to the proper writer
	PVCore::PVVersion _version;

	std::vector<std::string> _negated_terms;

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
	void parse_condition(rapidjson::Value const& obj) override;

	/** Generate between json
	 *
	 * {"range": {"id": {"gte" : begin, "lte": end}}}
	 */
	void between(rapidjson::Value const& id,
	             rapidjson::Value const& begin,
	             rapidjson::Value const& end) override;

	/** Generic comparison for less/greater/less_or_equal/greater_or_equal */
	void compare(rapidjson::Value const& id, rapidjson::Value const& end, const char*);

	/** Generate ends_with json
	 *
	 * {"regexp": {"id": "*value*"}}
	 */
	void contains(rapidjson::Value const& id, rapidjson::Value const& value) override;

	/** Generate is_empty json
	 *
	 * {"term": {"id": "value"}}
	 */
	void equal(rapidjson::Value const& id, rapidjson::Value const& value) override;

	/** Generate greater json
	 *
	 * {"range": {"id": {"gte" : "value"}}}
	 */
	void greater(rapidjson::Value const& id, rapidjson::Value const& end) override;

	/** Generate greater_or_equal json
	 *
	 * {"range": {"id": {"gt" : "value"}}}
	 */
	void greater_or_equal(rapidjson::Value const& id, rapidjson::Value const& end) override;

	/** Push id/value in the in_values dict for later processing.
	 */
	void in(rapidjson::Value const& id, rapidjson::Value const& values);

	/** Generate in json
	 *
	 * ForEach key, values in saved:
	 * {"terms": {"key": [values...]}}
	 */
	void inject_in(map_t const& saved) override;

	/** Generate is_empty json
	 *
	 * {"term": {"id": ""}}
	 */
	void is_empty(rapidjson::Value const& id) override;

	/** Generate less json
	 *
	 * {"range": {"id": {"lt" : "value"}}}
	 */
	void less(rapidjson::Value const& id, rapidjson::Value const& end) override;

	/** Generate less_or_equal json
	 *
	 * {"range": {"id": {"lte" : "value"}}}
	 */
	void less_or_equal(rapidjson::Value const& id, rapidjson::Value const& end) override;

	/** Push id/value in the not_in_values dict for later processing.
	 */
	void not_in(rapidjson::Value const& id, rapidjson::Value const& values);

	/** Generate begins_with json
	 *
	 * {"prefix": {"id": "value"}}
	 */
	void prefix(rapidjson::Value const& id, rapidjson::Value const& value) override;

	/** Generate ends_with json
	 *
	 * {"regexp": {"id": "value*"}}
	 */
	void suffix(rapidjson::Value const& id, rapidjson::Value const& value) override;

	/** Generic method to negate a node
	 *
	 * {"not": contains_condition }
	 */
	void pre_not_() override;
	void post_not_() override;

	/** Put "not" operations in a must_not block (ES 5.0.0+)
	 *
	 * {"must_not": [{"term":{"id1":"value1"}}, {"term":{"id2":"value2"}}, ...]}
	 */
	void inject_not(std::vector<std::string> const& saved);
};

#endif
