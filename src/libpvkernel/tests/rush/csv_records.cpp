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

// Where the records of a CSV end, and how its quoted values read.

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVCSVRecords.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

namespace
{

constexpr PVRush::PVCSVQuoting csv{',', '"'};

std::vector<std::string> records_of(const std::string& text)
{
	std::vector<const char*> ends;
	const char* const begin = text.data();
	const char* const end = begin + text.size();
	PV_VALID(PVRush::PVCSVRecords(csv).split(begin, end, false, ends), end, "text", text);

	std::vector<std::string> records;
	for (const char* record = begin; const char* record_end : ends) {
		records.emplace_back(record, record_end);
		record = record_end + 1;
	}
	return records;
}

void check_records(const std::string& text, const std::vector<std::string>& expected)
{
	const std::vector<std::string> records = records_of(text);
	PV_VALID(records.size(), expected.size(), "text", text);
	for (size_t i = 0; i < expected.size(); i++) {
		PV_VALID(records[i], expected[i], "text", text, "record", i);
	}
}

/**
 * Line feeds end the records but those quoted values hold, and quotes only quote a value
 * when they begin its field
 */
void check_record_ends()
{
	check_records("a,b\nc,d\n", {"a,b", "c,d"});
	check_records("\n\na\n", {"", "", "a"});
	check_records("a\n\"b\nc\"", {"a", "\"b\nc\""});
	check_records("1,\"x\ny\",2\n3,z,4\n", {"1,\"x\ny\",2", "3,z,4"});
	check_records("\"x\ny\",2\n", {"\"x\ny\",2"});
	check_records("1,\"x\"\"\ny\",2\n", {"1,\"x\"\"\ny\",2"});
	check_records("1,\"x\r\ny\",2\r\n3,z\r\n", {"1,\"x\r\ny\",2\r", "3,z\r"});
	check_records("1,\"\n\"\n2\n", {"1,\"\n\"", "2"});

	// quotes within a value are mere characters
	check_records("3,12\" pipe,4\n5,6\n", {"3,12\" pipe,4", "5,6"});
	check_records("9,key=\"a\nb\",10\n", {"9,key=\"a", "b\",10"});

	// nothing closes these opening quotes
	check_records("1,\"abc,2\n3,4\n", {"1,\"abc,2", "3,4"});
	check_records("1,\"abc\n2,\"x\",3\n", {"1,\"abc", "2,\"x\",3"});
	check_records("1,\"a\" b\nc\",2\n", {"1,\"a\" b", "c\",2"});
}

/**
 * A backslash only escapes the quotes of a value that RFC 4180 does not read
 */
void check_backslashes()
{
	// a Windows directory, then a line that has to stay on its own
	check_records("1,\"C:\\dir\\\",2\n3,4\n", {"1,\"C:\\dir\\\",2", "3,4"});
	check_records("\"C:\\dir\\\"\n3,4\n", {"\"C:\\dir\\\"", "3,4"});
	check_records("1,\"say \\\"hi\\\"\nok\",2\n3\n", {"1,\"say \\\"hi\\\"\nok\",2", "3"});

	const auto value_end = [](const std::string& value) {
		return PVRush::quoted_value_end(value.data(), value.data() + value.size(), csv, false);
	};
	using status = PVRush::PVQuotedValueEnd::status;

	const std::string windows = "C:\\dir\\\",2";
	const PVRush::PVQuotedValueEnd directory = value_end(windows);
	PV_VALID(directory.state == status::closed, true);
	PV_VALID(directory.quote - windows.data(), std::ptrdiff_t(7));
	PV_VALID(directory.backslash, false);

	const std::string escaped = "say \\\"hi\\\" ok\",2";
	const PVRush::PVQuotedValueEnd said = value_end(escaped);
	PV_VALID(said.state == status::closed, true);
	PV_VALID(said.quote - escaped.data(), std::ptrdiff_t(13));
	PV_VALID(said.backslash, true);

	// "a\"" as a program escaping with backslashes writes a"
	const std::string last = "a\\\"\",2";
	const PVRush::PVQuotedValueEnd ending = value_end(last);
	PV_VALID(ending.state == status::closed, true);
	PV_VALID(ending.quote - last.data(), std::ptrdiff_t(3));
	PV_VALID(ending.backslash, true);

	const std::string open = "abc,2";
	PV_VALID(value_end(open).state == status::not_closed, true);
	PV_VALID(PVRush::quoted_value_end(open.data(), open.data() + open.size(), csv, true).state ==
	             status::undecided,
	         true);
}

void check_unescape(std::string value, bool backslash, const std::string& expected)
{
	char* const begin = value.data();
	char* const end = PVRush::unescape_quoted_value(begin, begin + value.size(), '"', backslash);
	PV_VALID(std::string(begin, end), expected, "value", value);
}

void check_unescapes()
{
	check_unescape("He said \"\"hi\"\"", false, "He said \"hi\"");
	check_unescape("\"\"", false, "\"");
	check_unescape("C:\\dir\\", false, "C:\\dir\\");
	check_unescape("a\\\"\"b", false, "a\\\"b");
	check_unescape("say \\\"hi\\\"", true, "say \"hi\"");
	check_unescape("a\\\\\\\"b", true, "a\\\\\"b");
	check_unescape("a\\\"\"\"b", true, "a\"\"b");
	check_unescape("plain", true, "plain");
}

/**
 * A quoted value holds at most max_quoted_lines line feeds, past which its opening quote is
 * a mere character
 */
void check_line_limit()
{
	const size_t max = PVRush::PVCSVRecords::max_quoted_lines;
	const std::string longest = "1,\"" + std::string(max, '\n') + "\",2\n3\n";
	check_records(longest, {longest.substr(0, longest.size() - 3), "3"});

	const std::string too_long = "1,\"" + std::string(max + 1, '\n') + "\",2\n";
	const std::vector<std::string> records = records_of(too_long);
	PV_VALID(records.size(), max + 2);
	PV_VALID(records.front(), std::string("1,\""));
	PV_VALID(records.back(), std::string("\",2"));
}

/**
 * A CSV of @p count records as RFC 4180 writes them, and the records
 */
std::pair<std::string, std::vector<std::string>> written_csv(std::mt19937& random, size_t count)
{
	const std::string alphabet = "ab,\"\n\r\\ ";

	std::string text;
	std::vector<std::string> expected;
	for (size_t r = 0; r < count; r++) {
		std::string record;
		const size_t values = 1 + random() % 4;
		for (size_t v = 0; v < values; v++) {
			std::string value;
			for (size_t c = random() % 7; c != 0; c--) {
				value += alphabet[random() % alphabet.size()];
			}
			if (value.find_first_of(",\"\n\r") != std::string::npos) {
				std::string quoted = "\"";
				for (const char c : value) {
					quoted += c == '"' ? "\"\"" : std::string(1, c);
				}
				value = quoted + "\"";
			}
			record += (v == 0 ? "" : ",") + value;
		}
		expected.push_back(record);
		text += record + "\n";
	}
	return {text, expected};
}

/**
 * The records a CSV written as RFC 4180 tells read back whole, wherever the data read stops
 */
void check_written_records()
{
	std::mt19937 random(20261114);
	const auto [text, expected] = written_csv(random, 3000);
	check_records(text, expected);

	// the data read in pieces of random sizes, as a source reads chunks
	const PVRush::PVCSVRecords records(csv);
	std::vector<const char*> ends;
	const char* record = text.data();
	const char* read = text.data();
	const char* const text_end = text.data() + text.size();
	while (read != text_end) {
		read += std::min<size_t>(size_t(text_end - read), 1 + random() % 64);
		record = records.split(record, read, read != text_end, ends);
	}
	PV_VALID(ends.size(), expected.size());
	const char* begin = text.data();
	for (size_t i = 0; i < ends.size(); i++) {
		PV_VALID(std::string(begin, ends[i]), expected[i], "record", i);
		begin = ends[i] + 1;
	}
}

/**
 * The records split() reads 64 bytes at a time end where record_end() ends them, whatever
 * the quotes
 */
void check_blocks_as_record_end()
{
	const PVRush::PVCSVRecords records(csv);
	const auto check = [&records](const std::string& text) {
		const char* const end = text.data() + text.size();
		for (const bool more_data : {false, true}) {
			std::vector<const char*> ends;
			const char* const rest = records.split(text.data(), end, more_data, ends);

			std::vector<const char*> expected;
			const char* record = text.data();
			while (record != end) {
				const char* const record_end = records.record_end(record, end, more_data);
				if (record_end == nullptr) {
					break;
				}
				expected.push_back(record_end);
				record = record_end == end ? end : record_end + 1;
			}
			PV_VALID(ends == expected, true, "text", text, "more data", more_data);
			PV_VALID(rest, record, "text", text, "more data", more_data);
		}
	};

	std::mt19937 random(1114);
	for (const std::string alphabet : {"ab,\"\n", "a,\"\n\r\\ ", "\",\n", "abcdefgh,\""}) {
		for (size_t t = 0; t < 300; t++) {
			std::string text;
			for (size_t c = random() % 1500; c != 0; c--) {
				text += alphabet[random() % alphabet.size()];
			}
			check(text);
		}
	}

	// valid records but a few bytes
	for (size_t t = 0; t < 100; t++) {
		std::string text = written_csv(random, 50).first;
		for (size_t c = random() % 4; c != 0; c--) {
			text[random() % text.size()] = "\"\n,"[random() % 3];
		}
		check(text);
	}
}

} // namespace

int main()
{
	check_record_ends();
	check_backslashes();
	check_unescapes();
	check_line_limit();
	check_written_records();
	check_blocks_as_record_end();

	return 0;
}
