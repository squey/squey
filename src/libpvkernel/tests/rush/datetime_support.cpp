//
// MIT License
//
// © ESI Group, 2015
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

#include <pvkernel/rush/PVFormat.h>

#include <pvcop/db/array.h>
#include <pvcop/types/factory.h>
#include <pvcop/formatter_desc.h>

#include <iostream>

struct testcase_t {
	testcase_t(const std::string& f,
	           const std::string& tf,
	           const std::string& tsi,
	           const std::string& tso = "")
	    : formatter(f), time_format(tf), time_string_in(tsi)
	{
		time_string_out = tso.empty() ? tsi : tso;
	}

	std::string formatter;
	std::string time_format;
	std::string time_string_in;
	std::string time_string_out;
};

int main()
{
	std::vector<testcase_t> testcases;

	// libc
	testcases.emplace_back("datetime", "epoch", "1223884800");
	testcases.emplace_back("datetime", "epoch", "1334036784.745", "1334036784");
	testcases.emplace_back("datetime", "yyyy.MMMM.dd H:mm", "1996.Jul.10 12:08.32",
	                       "1996.Jul.10 12:08");
	testcases.emplace_back("datetime", "yyyy-M-d h:mm", "2017-03-19 12:08");
	testcases.emplace_back("datetime", "yyyy/M/d", "2014/11/07");
	testcases.emplace_back("datetime", "yyyy-M-d H:m:s", "2012-10-30 12:01:30");
	testcases.emplace_back("datetime", "MMM d H:m:s", "jul  3 23:59:58", "Jul 03 23:59:58");
	testcases.emplace_back("datetime", "MMM d yyyy H:m:s", "Jul 12 2013 00:03:15");
	testcases.emplace_back("datetime", "eee MMM d H:m:s yyyy", "Mon Sep 30 22:02:41 2013");
	testcases.emplace_back("datetime", "eee MMM d H:m:ss yyyy", "Mon Oct 13 10:00:00 2008");
	testcases.emplace_back("datetime", "eee MMM d 'S' H:m:ss yyyy", "Mon Oct 13 S 10:00:00 2008");
	testcases.emplace_back("datetime", "dMMMyyyy", "29Jan2012");
	testcases.emplace_back("datetime", "dMMMyyyy H:m:s", "8Apr2012 23:59:59", "08Apr2012 23:59:59");
	testcases.emplace_back("datetime", "d/M/yy H:m:s", "19/02/14 15:55:47");
	testcases.emplace_back("datetime", "H:m:s", "22:59:01");
	testcases.emplace_back("datetime", "H%m%s", "22%59%01");
	testcases.emplace_back("datetime", "yyyy/MM/dd HH:mm:ss Z", "2014/11/07 12:12:01 -0800",
	                       "2014/11/07 20:12:01 +0000");

	// boost
	testcases.emplace_back("datetime_us", "yyyy-M-d H:m:ss.S", "2017-03-19 10:00:59.001000");
	testcases.emplace_back("datetime_us", "yyyy-M-d H:m:ss.S", "2017-03-19 10:00:59.000000");
	testcases.emplace_back("datetime_us", "yyyy-M-d'T'H:m:ss.S", "2012-03-19T10:00:59.123",
	                       "2012-03-19T10:00:59.123000");
	testcases.emplace_back("datetime_us", "d/M/yyyy H:m:s.S", "19/02/2014 15:55:47.723000");
	testcases.emplace_back("datetime_us", "H:m:s.S", "05:35:02.506000");
	testcases.emplace_back("datetime_us", "H%m%s.S", "05%35%02.506000");

	// ICU
	testcases.emplace_back("datetime_ms", "epochS", "1452520190588");
	testcases.emplace_back("datetime_ms", "epoch.S", "1334036784.745");
	testcases.emplace_back("datetime_ms", "epoch.SSS", "1452654558.123");
	testcases.emplace_back("datetime_ms", "epoch.SSS", "1452654558.123456", "1452654558.123");
	testcases.emplace_back("datetime_ms", "epoch.S", "1334036784:745", "1334036784.745");
	testcases.emplace_back("datetime_ms", "epoch.S", "1334036784,745", "1334036784.745");
	testcases.emplace_back("datetime_ms", "yy-M-d H:mm:ss.SSS", "15-03-26 23:42:35.123",
	                       "15-3-26 23:42:35.123");
	testcases.emplace_back("datetime_ms", "dd-M-yy H:mm:ss:S", "19-02-14 15:55:47:123",
	                       "19-2-14 15:55:47:1");
	testcases.emplace_back("datetime_ms", "yy-M-d H:mm:ss.SSS V", "15-3-26 23:42:35.123 GMT",
	                       "15-3-26 23:42:35.123 gmt");

	// testcases.emplace_back("datetime_ms", "hh 'o''clock' a, zzzz", 	"12 o'clock PM, Pacific
	// Daylight Time"); // bug in ICU
	// testcases.emplace_back("datetime_ms", "K:mm a, z", 				"0:00 PM, PST");
	// //
	// bug
	// in
	// ICU
	// : http://bugs.icu-project.org/trac/ticket/11982
	// testcases.emplace_back("datetime_ms", "yyyy-M-dH:m:s.SZ",
	// "2015-3-2700:00:07.1882+01:00");
	// // bug in ICU : not deterministic
	// testcases.emplace_back("datetime_ms", "eee MMM d H:m:s V yyyy", 	"Tue Nov 9 13:11:46 EST
	// 2010");          // bug in ICU : wrong year

	for (const testcase_t& testcase : testcases) {

		const pvcop::formatter_desc& fd =
		    PVRush::PVFormat::get_datetime_formatter_desc(testcase.time_format);

		std::unique_ptr<pvcop::types::formatter_interface> fi(pvcop::types::factory::create(fd.name(), fd.parameters()));

		pvcop::db::array out_array(fi->name(), 1);

		const char* input_string = testcase.time_string_in.c_str();
		const char* output_string = testcase.time_string_out.c_str();
		if (fi->from_string(input_string, out_array.data(), 0)) { // from string

			static constexpr size_t str_len = pvcop::types::formatter_interface::MAX_STRING_LENGTH;
			char converted_string[str_len];

			if (fi->to_string(converted_string, str_len, out_array.data(), 0) > 0) { // to string
				if (strcmp(converted_string, output_string) != 0) {
					std::cerr << "Error: datetimes are not identical" << std::endl;
					std::cerr << "date: '" << output_string << "' - datetime format: '"
					          << testcase.time_format << "' - formatter parameters: '"
					          << fd.parameters() << "' - pvcop generate number: " << out_array.at(0)
					          << " - pvcop generated string: '" << converted_string << "'"
					          << std::endl;
					return 1;
				}

				if (fi->name() != testcase.formatter.c_str()) {
					std::cerr << "Formatter mismatch ! wanted: " << testcase.formatter
					          << ", used: " << fi->name() << std::endl;
					return 1;
				}

			} else {
				std::cerr << "Failed to format string " << std::endl;
				std::cerr << " date : " << output_string
				          << " - datetime format: " << testcase.time_format
				          << " - formatter parameters : " << fd.parameters()
				          << " - pvcop underlying value : " << out_array.at(0)
				          << " - pvcop generated string: " << converted_string << std::endl;
				return 1;
			}

		} else {
			std::cerr << "Failed to parse string" << std::endl;
			std::cerr << " date: " << input_string << " - datetime format: " << testcase.time_format
			          << " - formatter parameters: " << fd.parameters()
			          << " - pvcop underlying value: " << out_array.at(0) << std::endl;
			return 1;
		}
	}

	return 0;
}
