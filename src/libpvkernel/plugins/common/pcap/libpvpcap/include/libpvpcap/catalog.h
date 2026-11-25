/* * MIT License
 *
 * © Squey, 2026
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

#ifndef LIBPVPCAP_CATALOG_H
#define LIBPVPCAP_CATALOG_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pvpcap
{

/**
 * The protocols and the fields tshark knows, as `tshark -G` lists them: about 3,000 of the
 * first and 270,000 of the second.
 *
 * The columns of the packet list of Wireshark are fields of the protocol _ws.col, under the
 * names profiles give them: _ws.col.Source, _ws.col.Destination, _ws.col.Protocol and
 * _ws.col.Info.
 *
 * Thread-safety: none of its functions changes it.
 */
class field_catalog
{
  public:
	struct protocol {
		std::string_view name;        //!< such as "Hypertext Transfer Protocol"
		std::string_view short_name;  //!< such as "HTTP"
		std::string_view filter_name; //!< such as "http"
		std::vector<uint32_t> fields; //!< of fields(), by filter name
	};

	struct field {
		std::string_view name;        //!< such as "Host"
		std::string_view filter_name; //!< such as "http.host"
		std::string_view type;        //!< such as "FT_STRING"
		std::string_view description; //!< often empty
		uint32_t protocol = 0;        //!< of protocols()
	};

	/**
	 * Asks tshark, which takes it a few tenths of a second.
	 *
	 * @throw extraction_error if tshark can't be run
	 */
	static std::shared_ptr<const field_catalog> load();

	/**
	 * @param protocol_lines what `tshark -G protocols` writes
	 * @param field_lines what `tshark -G fields` writes
	 */
	field_catalog(std::vector<std::string> protocol_lines, std::vector<std::string> field_lines);

	field_catalog(const field_catalog&) = delete;
	field_catalog& operator=(const field_catalog&) = delete;

	const std::vector<protocol>& protocols() const { return _protocols; }
	const std::vector<field>& fields() const { return _fields; }

	//! Of fields(), -1 if tshark does not know it.
	int64_t find_field(std::string_view filter_name) const;
	//! Of protocols(), -1 if tshark does not know it.
	int64_t find_protocol(std::string_view filter_name) const;

	/**
	 * The fields whose filter name, name or description hold all the words of `query`, the
	 * best first: those whose filter name holds them, starting with them, short, of a common
	 * protocol, and those `bonus` favors.
	 *
	 * @param bonus to add to the score of a field of fields(), such as one seen in captures:
	 *              a word in the filter name is worth about 500
	 * @param matches set to how many fields hold the words, of which `most` are returned
	 */
	std::vector<uint32_t> search(std::string_view query,
	                             size_t most,
	                             const std::function<int(uint32_t)>& bonus = {},
	                             size_t* matches = nullptr) const;

  private:
	std::vector<std::string> _lines; //!< what the views of the protocols and fields point to
	std::vector<protocol> _protocols;
	std::vector<field> _fields;
	std::unordered_map<std::string_view, uint32_t> _protocols_by_name;
	std::unordered_map<std::string_view, uint32_t> _fields_by_name;
	//! For each field, its filter name, name and description in lower case, and a new line.
	std::string _haystack;
	//! Of each field in _haystack, and its end.
	std::vector<uint32_t> _starts;
	//! Of each field, whether its protocol is a common one.
	std::vector<bool> _common;
};

} // namespace pvpcap

#endif // LIBPVPCAP_CATALOG_H
