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

#ifndef __PVQUERYBUILDERJSONCONVERTER_H__
#define __PVQUERYBUILDERJSONCONVERTER_H__

#include <functional>
#include <unordered_map>
#include <string>
#include <vector>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

namespace PVCore
{

/** Interface to converter object from QueryBuilder json to database like input.
 *
 * @note A class is required to keep internal state about in/not_in operators
 *
 * @note We perform a compilation from one DSL to another. It could be done
 * creating an AST and visiting it. As the conversion is pretty easy and
 * simple (no optimisations or check) it is overkill...
 */
class PVQueryBuilderJsonConverter
{
  public:
	/** Parse json input to be processed
	 *
	 * @param qb_rule : json input
	 */
	explicit PVQueryBuilderJsonConverter(std::string const& qb_rule);

	/**
	 * Virtual destructor
	 */
	virtual ~PVQueryBuilderJsonConverter() = default;
	;

	/** Translate querybuilder json to database-like input.
	 *
	 * @return database-like input
	 */
	virtual std::string rules_to_json() = 0;

  protected:
	rapidjson::Document _doc; //!< json document with querybuilder json

	// We use a std::string as first parameter because const char* hash
	// use pointer as value.
	// const char* a = "ok", b = "ok" => std::hash(a) != std::hash(b)
	using map_t = std::unordered_map<std::string, std::vector<const char*>>;
	map_t _in_values;     //!< list of values with in operations for the current group
	map_t _not_in_values; //!< list of values with "not in" operations for the current group

  protected:
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
	virtual void parse_condition(rapidjson::Value const& obj) = 0;

	/** Generate between clause
	 */
	virtual void between(rapidjson::Value const& id,
	                     rapidjson::Value const& begin,
	                     rapidjson::Value const& end) = 0;

	/** Generate ends_with clause
	 */
	virtual void contains(rapidjson::Value const& id, rapidjson::Value const& value) = 0;

	/** Generate is_empty clause
	 */
	virtual void equal(rapidjson::Value const& id, rapidjson::Value const& value) = 0;

	/** Generate greater clause
	 */
	virtual void greater(rapidjson::Value const& id, rapidjson::Value const& end) = 0;

	/** Generate greater_or_equal clause
	 */
	virtual void greater_or_equal(rapidjson::Value const& id, rapidjson::Value const& end) = 0;

	/** Push id/value in the in_values dict for later processing.
	 */
	void in(rapidjson::Value const& id, rapidjson::Value const& value);

	/** Generate in clause
	 */
	virtual void inject_in(map_t const& saved) = 0;

	/** Generate is_empty clause
	 */
	virtual void is_empty(rapidjson::Value const& id) = 0;

	/** Generate less clause
	 */
	virtual void less(rapidjson::Value const& id, rapidjson::Value const& end) = 0;

	/** Generate less_or_equal clause
	 */
	virtual void less_or_equal(rapidjson::Value const& id, rapidjson::Value const& end) = 0;

	/** Push id/value in the not_in_values dict for later processing.
	 */
	void not_in(rapidjson::Value const& id, rapidjson::Value const& value);

	/** Generate begins_with clause
	 */
	virtual void prefix(rapidjson::Value const& id, rapidjson::Value const& value) = 0;

	/** Generate ends_with clause
	 */
	virtual void suffix(rapidjson::Value const& id, rapidjson::Value const& value) = 0;

	/** Generic method to negate a node
	 *
	 * @note : This method can't be virtual as it is also a template function.
	 * pre_not_ and post_not_ are used to generate correct information.
	 *
	 * Use an extra template parameter and a static_cast because the compiler doesn't understand
	 * inheritance dispatching with template and this is a PVQueryBuilderJsonConverter
	 * and not a derived pointer.
	 */
	template <class C, class F, class... T>
	void not_(F fun, T&&... params)
	{
		pre_not_();

		std::mem_fn(fun)(*static_cast<C*>(this), std::forward<T>(params)...);

		post_not_();
	}
	virtual void pre_not_() = 0;
	virtual void post_not_() = 0;

	/** Check if a json object is a condition node.
	 *
	 * Condition contains "condition" keyword.
	 */
	static bool is_condition(rapidjson::Value const& obj);
};
} // namespace PVCore

#endif
