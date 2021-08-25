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

#ifndef __PVSPLUNKJSONCONVERTER_H__
#define __PVSPLUNKJSONCONVERTER_H__

#include <string>
#include <sstream>

#include <pvkernel/core/PVQueryBuilderJsonConverter.h>

/** Converter object from QueryBuilder json to Splunk syntax
 */
class PVSplunkJsonConverter : public PVCore::PVQueryBuilderJsonConverter
{
  public:
	/** Parse json input to be processed
	 *
	 * @param qb_rule : json input
	 */
	PVSplunkJsonConverter(std::string const& qb_rule);

	/** Translate querybuilder json to splunk syntax
	 *
	 * @return splunk request input
	 */
	std::string rules_to_json();

  private:
	std::ostringstream _writer; //!< internal object to create splunk syntax

  private:
	/** Parse a condition and add it's infomations in the current request
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

	/** Generate between splunk request
	 *
	 * ( id<=end AND id>=begin )
	 */
	void
	between(rapidjson::Value const& id, rapidjson::Value const& begin, rapidjson::Value const& end);

	/** Generate ends_with splunk request
	 *
	 * id="*value*"
	 */
	void contains(rapidjson::Value const& id, rapidjson::Value const& value);

	/** Generate is_empty splunk request
	 *
	 * id="value"
	 */
	void equal(rapidjson::Value const& id, rapidjson::Value const& value);

	/** Generate greater splunk request
	 *
	 * id>=value
	 */
	void greater(rapidjson::Value const& id, rapidjson::Value const& end);

	/** Generate greater_or_equal splunk request
	 *
	 * id>value
	 */
	void greater_or_equal(rapidjson::Value const& id, rapidjson::Value const& end);

	/** Generate in splunk request
	 *
	 * ( id=value1 OR id=value2 OR ...)
	 */
	void inject_in(map_t const& saved);

	/** Generate is_empty splunk request
	 *
	 * id=""
	 */
	void is_empty(rapidjson::Value const& /*id*/) { assert("not implemented yet" && false); }

	/** Generate less splunk request
	 *
	 * id<value
	 */
	void less(rapidjson::Value const& id, rapidjson::Value const& end);

	/** Generate less_or_equal splunk request
	 *
	 * id<=value
	 */
	void less_or_equal(rapidjson::Value const& id, rapidjson::Value const& end);

	/** Generate begins_with splunk request
	 *
	 * id="value*"
	 */
	void prefix(rapidjson::Value const& id, rapidjson::Value const& value);

	/** Generate ends_with splunk request
	 *
	 * id="value*"
	 */
	void suffix(rapidjson::Value const& id, rapidjson::Value const& value);

	/** Generic method to negate a node
	 *
	 * NOT ( content...)
	 */
	void pre_not_();
	void post_not_();
};

#endif
