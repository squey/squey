//
// MIT License
//
// © Squey, 2026
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

#include "../include/libpvpcap/catalog.h"

#include "../include/libpvpcap/extraction.h"
#include "../include/libpvpcap/shell.h"
#include "../include/libpvpcap/ws.h"

#include <QFile>
#include <QString>

#include <algorithm>
#include <array>
#include <cctype>

namespace pvpcap
{

namespace
{

std::vector<std::string_view> split_tabs(std::string_view line)
{
	std::vector<std::string_view> columns;
	size_t begin = 0;
	while (true) {
		const size_t end = line.find('\t', begin);
		columns.push_back(line.substr(begin, end == std::string_view::npos ? end : end - begin));
		if (end == std::string_view::npos) {
			return columns;
		}
		begin = end + 1;
	}
}

char lower(char c)
{
	return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

bool word_start(std::string_view text, size_t position)
{
	return position == 0 or not std::isalnum(static_cast<unsigned char>(text[position - 1]));
}

//! Of protocols most captures hold, whose fields come first in searches.
constexpr std::array<std::string_view, 20> COMMON_PROTOCOLS = {
    "_ws.col", "arp",  "dhcp", "dns",  "eth", "frame", "http", "http2", "icmp", "icmpv6",
    "ip",      "ipv6", "ntp",  "quic", "sip", "smb2",  "ssh",  "tcp",   "tls",  "udp"};

//! What `tshark -G protocols` says of the columns, in case it does not.
const std::string COLUMNS_PROTOCOL_LINE = "Wireshark Columns\tColumns\t_ws.col";

//! The columns of the packet list of Wireshark, as `tshark -G fields` would list them.
const std::array<std::string, 4> COLUMN_LINES = {
    "F\tSource\t_ws.col.Source\tFT_STRING\t_ws.col\t\t0x0\tSource address of the packet, "
    "whatever its protocol, as the packet list of Wireshark shows it",
    "F\tDestination\t_ws.col.Destination\tFT_STRING\t_ws.col\t\t0x0\tDestination address of "
    "the packet, whatever its protocol, as the packet list of Wireshark shows it",
    "F\tProtocol\t_ws.col.Protocol\tFT_STRING\t_ws.col\t\t0x0\tThe highest protocol of the "
    "packet, as the packet list of Wireshark shows it",
    "F\tInfo\t_ws.col.Info\tFT_STRING\t_ws.col\t\t0x0\tWhat the packet holds, in a few words, "
    "as the packet list of Wireshark shows it"};

} // namespace

std::shared_ptr<const field_catalog> field_catalog::load()
{
	// Earlier versions kept the fields of tshark in this file, which nothing reads anymore.
	QFile::remove(QString::fromStdString(get_user_conf_dir() + "/ws_protocols_dict"));

	std::vector<std::string> fields = ws_get_tshark_fields();
	if (fields.empty()) {
		throw extraction_error("tshark could not be run to list the fields it knows.");
	}
	return std::make_shared<const field_catalog>(ws_get_tshark_protocols(), std::move(fields));
}

field_catalog::field_catalog(std::vector<std::string> protocol_lines,
                             std::vector<std::string> field_lines)
{
	// The views point into _lines, which is not changed once they are made.
	_lines = std::move(protocol_lines);
	_lines.push_back(COLUMNS_PROTOCOL_LINE);
	const size_t protocols_end = _lines.size();
	_lines.reserve(_lines.size() + field_lines.size() + COLUMN_LINES.size());
	std::move(field_lines.begin(), field_lines.end(), std::back_inserter(_lines));
	const size_t columns_begin = _lines.size();
	_lines.insert(_lines.end(), COLUMN_LINES.begin(), COLUMN_LINES.end());

	for (size_t i = 0; i < protocols_end; i++) {
		const std::vector<std::string_view> columns = split_tabs(_lines[i]);
		if (columns.size() < 3 or columns[2].empty()) {
			continue;
		}
		if (_protocols_by_name.try_emplace(columns[2], _protocols.size()).second) {
			_protocols.push_back({columns[0], columns[1], columns[2], {}});
		}
	}

	for (size_t i = protocols_end; i < _lines.size(); i++) {
		const std::string_view line = _lines[i];
		if (not line.starts_with("F\t")) {
			continue;
		}
		const std::vector<std::string_view> columns = split_tabs(line);
		if (columns.size() < 5) {
			continue;
		}
		const std::string_view filter_name = columns[2];
		// The other columns repeat fields, or are aliases of those of COLUMN_LINES.
		if (filter_name.starts_with("_ws.col.") and i < columns_begin) {
			continue;
		}
		const auto protocol = _protocols_by_name.find(columns[4]);
		if (protocol == _protocols_by_name.end() or
		    not _fields_by_name.try_emplace(filter_name, _fields.size()).second) {
			continue;
		}
		_fields.push_back({columns[1], filter_name, columns[3],
		                   columns.size() > 7 ? columns[7] : std::string_view(), protocol->second});
		_protocols[protocol->second].fields.push_back(uint32_t(_fields.size() - 1));
	}

	for (protocol& p : _protocols) {
		std::sort(p.fields.begin(), p.fields.end(), [this](uint32_t a, uint32_t b) {
			return _fields[a].filter_name < _fields[b].filter_name;
		});
	}

	_starts.reserve(_fields.size() + 1);
	_common.reserve(_fields.size());
	for (const field& f : _fields) {
		_starts.push_back(uint32_t(_haystack.size()));
		for (const std::string_view part : {f.filter_name, f.name, f.description}) {
			std::transform(part.begin(), part.end(), std::back_inserter(_haystack), lower);
			_haystack.push_back('\t');
		}
		_haystack.back() = '\n';
		const std::string_view protocol = _protocols[f.protocol].filter_name;
		_common.push_back(std::find(COMMON_PROTOCOLS.begin(), COMMON_PROTOCOLS.end(), protocol) !=
		                  COMMON_PROTOCOLS.end());
	}
	_starts.push_back(uint32_t(_haystack.size()));
}

int64_t field_catalog::find_field(std::string_view filter_name) const
{
	const auto it = _fields_by_name.find(filter_name);
	return it != _fields_by_name.end() ? int64_t(it->second) : -1;
}

int64_t field_catalog::find_protocol(std::string_view filter_name) const
{
	const auto it = _protocols_by_name.find(filter_name);
	return it != _protocols_by_name.end() ? int64_t(it->second) : -1;
}

std::vector<uint32_t> field_catalog::search(std::string_view query,
                                            size_t most,
                                            const std::function<int(uint32_t)>& bonus,
                                            size_t* matches) const
{
	if (matches != nullptr) {
		*matches = 0;
	}
	std::vector<std::string> words(1);
	for (const char c : query) {
		if (std::isspace(static_cast<unsigned char>(c))) {
			if (not words.back().empty()) {
				words.emplace_back();
			}
		} else {
			words.back().push_back(lower(c));
		}
	}
	if (words.back().empty()) {
		words.pop_back();
	}
	if (words.empty() or most == 0) {
		return {};
	}
	std::string whole = words.front();
	for (size_t i = 1; i < words.size(); i++) {
		whole += " " + words[i];
	}
	// The longest word is the rarest: it finds the fields, which the others are then looked
	// for in.
	const std::string& key = *std::max_element(
	    words.begin(), words.end(),
	    [](const std::string& a, const std::string& b) { return a.size() < b.size(); });

	std::vector<std::pair<int64_t, uint32_t>> scored;
	size_t at = 0;
	while ((at = _haystack.find(key, at)) != std::string::npos) {
		const auto i =
		    uint32_t(std::upper_bound(_starts.begin(), _starts.end(), at) - _starts.begin() - 1);
		at = _starts[i + 1];
		const field& f = _fields[i];
		const std::string_view text(_haystack.data() + _starts[i], _starts[i + 1] - _starts[i]);
		const std::string_view filter_name = text.substr(0, f.filter_name.size());
		const std::string_view name = text.substr(f.filter_name.size() + 1, f.name.size());

		int64_t score = 0;
		bool found = true;
		for (const std::string& word : words) {
			if (const size_t p = filter_name.find(word); p != std::string_view::npos) {
				score += 500 + (word_start(filter_name, p) ? 300 : 0);
			} else if (const size_t q = name.find(word); q != std::string_view::npos) {
				score += 250 + (word_start(name, q) ? 150 : 0);
			} else if (text.find(word) != std::string_view::npos) {
				score += 40;
			} else {
				found = false;
				break;
			}
		}
		if (not found) {
			continue;
		}
		if (filter_name == whole) {
			score += 10000;
		} else if (filter_name.starts_with(whole)) {
			score += 2000;
		}
		score += _common[i] ? 400 : 0;
		score -= int64_t(f.filter_name.size()) * 4;
		if (bonus) {
			score += bonus(i);
		}
		scored.emplace_back(score, i);
	}

	if (matches != nullptr) {
		*matches = scored.size();
	}
	const size_t count = std::min(most, scored.size());
	std::partial_sort(
	    scored.begin(), scored.begin() + count, scored.end(), [this](const auto& a, const auto& b) {
		    return a.first != b.first
		               ? a.first > b.first
		               : _fields[a.second].filter_name < _fields[b.second].filter_name;
	    });
	std::vector<uint32_t> found(count);
	std::transform(scored.begin(), scored.begin() + count, found.begin(),
	               [](const auto& s) { return s.second; });
	return found;
}

} // namespace pvpcap
