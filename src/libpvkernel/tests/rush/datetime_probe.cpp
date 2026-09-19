//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

// Probe: which time library reads which format correctly, and how fast.
//   probe                 reads every case with the formatter chosen for it, then with each
//                         library that could read it
//   probe bench [N] [R]   times reading and writing N values with each library, keeping the
//                         best of R rounds, run in alternate orders
//   probe leak            reads a date without zone before and after one with a zone
//   probe select          prints the formatter chosen for each pattern read on stdin
//
// libc_us_prototype is not used by Squey: it measures what a microsecond formatter built on
// strptime and strftime, the fraction and the offset read by hand, would cost.

#include <pvkernel/rush/PVFormat.h>

#include <pvcop/db/array.h>
#include <pvcop/formatter_desc.h>
#include <pvcop/types/datetime_us.h>
#include <pvcop/types/factory.h>

#include <boost/date_time/posix_time/posix_time.hpp>

#include <tbb/parallel_for.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace
{

using pt = boost::posix_time::ptime;

struct expected_t {
	int y, mo, d, h, mi, s, us;
	int offset_minutes; // offset of the input string, the value is expected in UTC
};

struct case_t {
	std::string icu;
	std::string input;
	std::optional<expected_t> expected;
	std::string libc;  // strptime parameters, empty if not applicable
	std::string boost; // boost parameters, empty if not applicable
	std::string note;
};

int64_t epoch_us(const expected_t& e)
{
	const pt t(boost::gregorian::date(e.y, e.mo, e.d),
	           boost::posix_time::time_duration(e.h, e.mi, e.s) +
	               boost::posix_time::microseconds(e.us) - boost::posix_time::minutes(e.offset_minutes));
	return (t - pt(boost::gregorian::date(1970, 1, 1))).total_microseconds();
}

int64_t to_epoch_us(const std::string& name, uint64_t raw)
{
	if (name == "datetime") {
		return int64_t(raw) * 1000000;
	}
	if (name == "datetime_ms") {
		return int64_t(raw) * 1000;
	}
	const pt t = pvcop::types::formatter_datetime_us::cal(raw).as_time;
	return (t - pt(boost::gregorian::date(1970, 1, 1))).total_microseconds();
}

std::string show_us(int64_t v)
{
	const pt t = pt(boost::gregorian::date(1970, 1, 1)) + boost::posix_time::microseconds(v);
	return boost::posix_time::to_iso_extended_string(t);
}

void check(const std::string& label,
           const std::string& name,
           const std::string& params,
           const case_t& c)
{
	std::cout << "    " << std::left << std::setw(8) << label << std::setw(12) << name << std::setw(34)
	          << ("'" + params + "'");
	std::unique_ptr<pvcop::types::formatter_interface> fi;
	try {
		fi.reset(pvcop::types::factory::create(name, params));
	} catch (const std::exception& e) {
		std::cout << "cannot create: " << e.what() << "\n";
		return;
	}
	pvcop::db::array a(fi->name(), 1);
	std::memset(a.data(), 0, sizeof(uint64_t));
	bool ok = false;
	try {
		ok = fi->from_string(c.input.c_str(), a.data(), 0);
	} catch (const std::exception& e) {
		std::cout << "THROWS " << e.what() << "\n";
		return;
	}
	if (not ok) {
		std::cout << "REFUSED\n";
		return;
	}
	const uint64_t raw = *reinterpret_cast<const uint64_t*>(a.data());
	const int64_t got = to_epoch_us(name, raw);
	char out[256] = {};
	const int n = fi->to_string(out, sizeof(out), a.data(), 0);
	std::cout << show_us(got);
	if (c.expected) {
		const int64_t want = epoch_us(*c.expected);
		if (got == want) {
			std::cout << "  exact";
		} else {
			std::cout << "  OFF BY " << (got - want) << " us";
		}
	}
	std::cout << "  -> '" << (n >= 0 ? std::string(out) : std::string("<to_string ") + std::to_string(n) + ">")
	          << "'\n";
}

std::vector<case_t> cases()
{
	return {
	    // baseline
	    {"yyyy-MM-dd HH:mm:ss", "2014-11-07 12:12:01", expected_t{2014, 11, 7, 12, 12, 1, 0, 0},
	     "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "baseline"},
	    {"yyyy-MM-dd HH:mm:ss.SSSSSS", "2014-11-07 12:12:01.123456",
	     expected_t{2014, 11, 7, 12, 12, 1, 123456, 0}, "", "%Y-%m-%d %H:%M:%S.%f", "fraction"},
	    // zone + fraction
	    {"yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'", "2026-01-14T00:06:46.532856Z",
	     expected_t{2026, 1, 14, 0, 6, 46, 532856, 0}, "", "%Y-%m-%dT%H:%M:%S.%fZ", "literal Z"},
	    {"yyyy-MM-dd'T'HH:mm:ss.SSSSSSX", "2026-01-14T00:06:46.532856Z",
	     expected_t{2026, 1, 14, 0, 6, 46, 532856, 0}, "", "", "ISO zone X, UTC"},
	    {"yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX", "2026-01-14T02:06:46.532856+02:00",
	     expected_t{2026, 1, 14, 2, 6, 46, 532856, 120}, "", "", "ISO zone XXX, +02:00"},
	    {"yyyy-MM-dd HH:mm:ss.SSS Z", "2014-11-07 12:12:01.123 -0800",
	     expected_t{2014, 11, 7, 12, 12, 1, 123000, -480}, "", "", "RFC zone Z + fraction"},
	    // zone without fraction: libc today
	    {"yyyy/MM/dd HH:mm:ss Z", "2014/11/07 12:12:01 -0800",
	     expected_t{2014, 11, 7, 12, 12, 1, 0, -480}, "%Y/%m/%d %H:%M:%S %z", "", "RFC zone Z"},
	    {"yyyy-MM-dd HH:mm:ss ZZ", "2014-11-07 12:12:01 -0800",
	     expected_t{2014, 11, 7, 12, 12, 1, 0, -480}, "%Y-%m-%d %H:%M:%S %z", "", "RFC zone ZZ"},
	    {"yyyy-MM-dd HH:mm:ss X", "2014-11-07 12:12:01 -08",
	     expected_t{2014, 11, 7, 12, 12, 1, 0, -480}, "%Y-%m-%d %H:%M:%S %z", "", "ISO zone X"},
	    {"yyyy-MM-dd HH:mm:ss XX", "2014-11-07 12:12:01 -0800",
	     expected_t{2014, 11, 7, 12, 12, 1, 0, -480}, "%Y-%m-%d %H:%M:%S %z", "", "ISO zone XX"},
	    {"yyyy-MM-dd HH:mm:ss xx", "2014-11-07 12:12:01 -0800",
	     expected_t{2014, 11, 7, 12, 12, 1, 0, -480}, "%Y-%m-%d %H:%M:%S %z", "", "ISO zone xx"},
	    {"yyyy-MM-dd'T'HH:mm:ssX", "2014-11-07T12:12:01Z",
	     expected_t{2014, 11, 7, 12, 12, 1, 0, 0}, "%Y-%m-%dT%H:%M:%S%z", "", "ISO zone X, UTC"},
	    {"yyyy-MM-dd HH:mm:ss XXX", "2014-11-07 12:12:01 -08:00",
	     expected_t{2014, 11, 7, 12, 12, 1, 0, -480}, "%Y-%m-%d %H:%M:%S %z", "", "ISO zone XXX"},
	    // fraction not preceded by a dot
	    {"yyyy-MM-dd HH:mm:ss,SSS", "2016-01-01 12:00:00,123",
	     expected_t{2016, 1, 1, 12, 0, 0, 123000, 0}, "", "%Y-%m-%d %H:%M:%S,%f", "log4j comma"},
	    // two digit years
	    {"d/M/yy H:m:s.S", "19/02/14 15:55:47.723000", expected_t{2014, 2, 19, 15, 55, 47, 723000, 0},
	     "", "%d/%m/%y %H:%M:%S.%f", "yy=14"},
	    {"d/M/yy H:m:s.S", "19/02/85 15:55:47.723000", expected_t{1985, 2, 19, 15, 55, 47, 723000, 0},
	     "", "%d/%m/%y %H:%M:%S.%f", "yy=85"},
	    {"d/M/yy H:m:s", "19/02/85 15:55:47", expected_t{1985, 2, 19, 15, 55, 47, 0, 0},
	     "%d/%m/%y %H:%M:%S", "%d/%m/%y %H:%M:%S", "yy=85, no fraction"},
	    {"d/M/yy H:m:s", "19/02/45 15:55:47", expected_t{2045, 2, 19, 15, 55, 47, 0, 0},
	     "%d/%m/%y %H:%M:%S", "%d/%m/%y %H:%M:%S", "yy=45, no fraction"},
	    // 12 hour clock
	    {"yyyy-M-d h:mm:ss.S a", "2017-03-19 1:08:07.123000 PM",
	     expected_t{2017, 3, 19, 13, 8, 7, 123000, 0}, "", "%Y-%m-%d %I:%M:%S.%f %p", "12h + fraction"},
	    {"yyyy-M-d h:mm a", "2017-03-19 1:08 PM", expected_t{2017, 3, 19, 13, 8, 0, 0, 0},
	     "%Y-%m-%d %I:%M %p", "", "12h"},
	    {"yyyy-MM-dd K:mm a", "2014-11-07 1:08 PM", expected_t{2014, 11, 7, 13, 8, 0, 0, 0},
	     "%Y-%m-%d %I:%M %p", "", "hour K (0-11)"},
	    // epoch
	    {"epoch", "1334036784", expected_t{2012, 4, 10, 5, 46, 24, 0, 0}, "%s", "", "epoch"},
	    {"epoch.S", "1334036784.745123", expected_t{2012, 4, 10, 5, 46, 24, 745123, 0}, "", "",
	     "epoch with microseconds"},
	    {"epoch.SSSSSS", "1334036784.745123", expected_t{2012, 4, 10, 5, 46, 24, 745123, 0}, "", "",
	     "epoch with microseconds"},
	    // ICU bugs noted in datetime_support.cpp in 2016
	    {"hh 'o''clock' a, zzzz", "12 o'clock PM, Pacific Daylight Time",
	     expected_t{1970, 1, 1, 12, 0, 0, 0, -420}, "", "", "ICU bug 2016"},
	    {"K:mm a, z", "0:00 PM, PST", expected_t{1970, 1, 1, 12, 0, 0, 0, -480}, "", "",
	     "ICU ticket 11982"},
	    {"yyyy-M-dH:m:s.SZ", "2015-3-2700:00:07.1882+01:00",
	     expected_t{2015, 3, 27, 0, 0, 7, 188000, 60}, "", "", "ICU non deterministic"},
	    {"eee MMM d H:m:s V yyyy", "Tue Nov 9 13:11:46 EST 2010",
	     expected_t{2010, 11, 9, 13, 11, 46, 0, -300}, "", "", "ICU wrong year"},
	    {"eee MMM d H:m:s z yyyy", "Tue Nov 9 13:11:46 EST 2010",
	     expected_t{2010, 11, 9, 13, 11, 46, 0, -300}, "", "", "zone abbreviation"},
	};
}

int probe()
{
	for (const case_t& c : cases()) {
		std::cout << "[" << c.note << "] '" << c.icu << "'  <-  '" << c.input << "'\n";
		const pvcop::formatter_desc fd = PVRush::PVFormat::get_datetime_formatter_desc(c.icu);
		check("today", fd.name(), fd.parameters(), c);
		if (not c.libc.empty()) {
			check("libc", "datetime", c.libc, c);
		}
		if (not c.boost.empty()) {
			check("boost", "datetime_us", c.boost, c);
		}
		check("ICU", "datetime_ms", c.icu, c);
	}
	return 0;
}

struct bench_case_t {
	std::string label;
	std::string name;
	std::string params;
	std::string strftime_like; // how the input strings are generated
	bool with_fraction;
	std::string suffix;
};

std::vector<std::string> generate(size_t n, const bench_case_t& b)
{
	std::vector<std::string> v(n);
	std::mt19937_64 gen(42);
	std::uniform_int_distribution<int64_t> dist(0, int64_t(2000000000) * 1000000);
	for (auto& s : v) {
		const int64_t us = dist(gen);
		const pt t = pt(boost::gregorian::date(1970, 1, 1)) + boost::posix_time::microseconds(us);
		char buf[128];
		const auto d = t.date();
		const auto tod = t.time_of_day();
		if (b.strftime_like == "iso") {
			std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d", int(d.year()),
			              int(d.month()), int(d.day()), int(tod.hours()), int(tod.minutes()),
			              int(tod.seconds()));
		} else { // "isoT"
			std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d", int(d.year()),
			              int(d.month()), int(d.day()), int(tod.hours()), int(tod.minutes()),
			              int(tod.seconds()));
		}
		s = buf;
		if (b.with_fraction) {
			std::snprintf(buf, sizeof(buf), ".%06d", int(tod.fractional_seconds()));
			s += buf;
		}
		s += b.suffix;
	}
	return v;
}

/**
 * Prototype of a microsecond formatter built on strptime/strftime: the format is split
 * around its fraction, each part goes through the libc, the fraction is read by hand
 */
struct libc_us_prototype {
	std::string before; // up to the seconds, the separator excluded
	char separator = 0; // what precedes the fraction in the strings
	std::string after;  // what follows the fraction

	explicit libc_us_prototype(const std::string& params)
	{
		const size_t f = params.find("%f");
		before = params.substr(0, f);
		if (not before.empty() and before.back() != 'S') {
			separator = before.back();
			before.pop_back();
		}
		after = params.substr(f + 2);
	}

	static int64_t days_from_civil(int64_t y, unsigned m, unsigned d)
	{
		y -= m <= 2;
		const int64_t era = (y >= 0 ? y : y - 399) / 400;
		const unsigned yoe = unsigned(y - era * 400);
		const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
		const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
		return era * 146097 + int64_t(doe) - 719468;
	}

	bool parse(const char* s, int64_t& us) const
	{
		tm t{};
		const char* p = strptime(s, before.c_str(), &t);
		if (not p) {
			return false;
		}
		if (separator) {
			if (*p != separator) {
				return false;
			}
			++p;
		}
		int64_t frac = 0;
		int digits = 0;
		for (; *p >= '0' and *p <= '9'; ++p, ++digits) {
			if (digits < 6) {
				frac = frac * 10 + (*p - '0');
			}
		}
		if (digits == 0) {
			return false;
		}
		for (int i = digits; i < 6; i++) {
			frac *= 10;
		}
		if (not after.empty() and not strptime(p, after.c_str(), &t)) {
			return false;
		}
		const int64_t days = days_from_civil(t.tm_year + 1900, t.tm_mon + 1, t.tm_mday);
		us = (days * 86400 + t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec - t.tm_gmtoff) * 1000000 +
		     frac;
		return true;
	}

	int format(char* out, size_t len, int64_t us) const
	{
		const int64_t secs = us >= 0 ? us / 1000000 : (us - 999999) / 1000000;
		const int frac = int(us - secs * 1000000);
		// civil_from_days
		const int64_t z0 = (secs >= 0 ? secs : secs - 86399) / 86400;
		const int64_t sod = secs - z0 * 86400;
		const int64_t z = z0 + 719468;
		const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
		const unsigned doe = unsigned(z - era * 146097);
		const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
		const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
		const unsigned mp = (5 * doy + 2) / 153;
		const unsigned d = doy - (153 * mp + 2) / 5 + 1;
		const unsigned m = mp < 10 ? mp + 3 : mp - 9;
		tm t{};
		t.tm_year = int(int64_t(yoe) + era * 400 + (m <= 2)) - 1900;
		t.tm_mon = int(m) - 1;
		t.tm_mday = int(d);
		t.tm_hour = int(sod / 3600);
		t.tm_min = int(sod / 60 % 60);
		t.tm_sec = int(sod % 60);
		t.tm_wday = int((z0 + 4) % 7 + 7) % 7;
		size_t n = std::strftime(out, len, before.c_str(), &t);
		if (separator) {
			out[n++] = separator;
		}
		n += std::snprintf(out + n, len - n, "%06d", frac);
		if (not after.empty()) {
			n += std::strftime(out + n, len - n, after.c_str(), &t);
		}
		return int(n);
	}
};

double time_it(const std::function<void()>& f)
{
	const auto start = std::chrono::steady_clock::now();
	f();
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

int bench(size_t n, int rounds)
{
	const std::vector<bench_case_t> benches = {
	    {"seconds", "datetime", "%Y-%m-%d %H:%M:%S", "iso", false, ""},
	    {"seconds", "datetime_us", "%Y-%m-%d %H:%M:%S", "iso", false, ""},
	    {"seconds", "datetime_ms", "yyyy-MM-dd HH:mm:ss", "iso", false, ""},
	    {"micro", "datetime_us", "%Y-%m-%d %H:%M:%S.%f", "iso", true, ""},
	    {"micro", "datetime_ms", "yyyy-MM-dd HH:mm:ss.SSSSSS", "iso", true, ""},
	    {"microZ", "datetime_us", "%Y-%m-%dT%H:%M:%S.%fZ", "isoT", true, "Z"},
	    {"microZ", "datetime_ms", "yyyy-MM-dd'T'HH:mm:ss.SSSSSSX", "isoT", true, "Z"},
	    {"secZ+2", "datetime", "%Y-%m-%dT%H:%M:%S%z", "isoT", false, "+02:00"},
	    {"secZ+2", "datetime_ms", "yyyy-MM-dd'T'HH:mm:ssXXX", "isoT", false, "+02:00"},
	    {"micZ+2", "datetime_ms", "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX", "isoT", true, "+02:00"},
	};

	struct result_t {
		double parse_seq = 1e9, parse_par = 1e9, format_par = 1e9;
	};
	std::vector<result_t> results(benches.size());

	std::vector<std::vector<std::string>> inputs;
	for (const auto& b : benches) {
		inputs.push_back(generate(n, b));
	}

	for (int r = 0; r < rounds; r++) {
		// alternate the order from one round to the next, see the measurement protocol
		std::vector<size_t> order(benches.size());
		std::iota(order.begin(), order.end(), 0);
		if (r % 2) {
			std::reverse(order.begin(), order.end());
		}
		for (size_t i : order) {
			const auto& b = benches[i];
			const auto& in = inputs[i];
			std::unique_ptr<pvcop::types::formatter_interface> fi(
			    pvcop::types::factory::create(b.name, b.params));
			pvcop::db::array a(fi->name(), n);
			size_t refused = 0;

			const double seq = time_it([&]() {
				for (size_t k = 0; k < n; k++) {
					refused += not fi->from_string(in[k].c_str(), a.data(), k);
				}
			});
			const double par = time_it([&]() {
				tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const auto& range) {
					for (size_t k = range.begin(); k != range.end(); k++) {
						(void)fi->from_string(in[k].c_str(), a.data(), k);
					}
				});
			});
			const double fmt = time_it([&]() {
				tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const auto& range) {
					char buf[128];
					for (size_t k = range.begin(); k != range.end(); k++) {
						(void)fi->to_string(buf, sizeof(buf), a.data(), k);
					}
				});
			});
			if (refused) {
				std::cerr << b.label << " " << b.name << ": " << refused << " values refused\n";
			}
			results[i].parse_seq = std::min(results[i].parse_seq, seq);
			results[i].parse_par = std::min(results[i].parse_par, par);
			results[i].format_par = std::min(results[i].format_par, fmt);
		}
	}

	// the prototype, on the same strings as the formatters
	const std::vector<std::pair<size_t, std::string>> prototypes = {
	    {3, "%Y-%m-%d %H:%M:%S.%f"}, {5, "%Y-%m-%dT%H:%M:%S.%f%z"}};
	std::vector<result_t> proto_results(prototypes.size());
	for (int r = 0; r < rounds; r++) {
		for (size_t j = 0; j < prototypes.size(); j++) {
			const libc_us_prototype proto(prototypes[j].second);
			const auto& in = inputs[prototypes[j].first];
			std::vector<int64_t> values(n);
			size_t refused = 0;
			const double seq = time_it([&]() {
				for (size_t k = 0; k < n; k++) {
					refused += not proto.parse(in[k].c_str(), values[k]);
				}
			});
			const double par = time_it([&]() {
				tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const auto& range) {
					for (size_t k = range.begin(); k != range.end(); k++) {
						(void)proto.parse(in[k].c_str(), values[k]);
					}
				});
			});
			std::atomic<size_t> mismatches = 0;
			const double fmt = time_it([&]() {
				tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const auto& range) {
					char buf[128];
					for (size_t k = range.begin(); k != range.end(); k++) {
						(void)proto.format(buf, sizeof(buf), values[k]);
					}
				});
			});
			// round trip, outside of the timing
			for (size_t k = 0; k < n; k += 997) {
				char buf[128];
				proto.format(buf, sizeof(buf), values[k]);
				int64_t back = 0;
				if (not proto.parse(buf, back) or back != values[k]) {
					++mismatches;
				}
			}
			if (refused or mismatches) {
				std::cerr << "prototype " << prototypes[j].second << ": " << refused
				          << " refused, " << mismatches << " round trip mismatches\n";
			}
			proto_results[j].parse_seq = std::min(proto_results[j].parse_seq, seq);
			proto_results[j].parse_par = std::min(proto_results[j].parse_par, par);
			proto_results[j].format_par = std::min(proto_results[j].format_par, fmt);
		}
	}

	std::cout << std::left << std::setw(8) << "values" << std::setw(13) << "formatter"
	          << std::right << std::setw(14) << "parse 1 thr" << std::setw(14) << "parse N thr"
	          << std::setw(14) << "format N thr" << "   (ns per value, min of " << rounds
	          << " rounds)\n";
	for (size_t i = 0; i < benches.size(); i++) {
		const auto& b = benches[i];
		const auto& res = results[i];
		std::cout << std::left << std::setw(8) << b.label << std::setw(13) << b.name << std::right
		          << std::fixed << std::setprecision(1) << std::setw(14) << res.parse_seq * 1e9 / n
		          << std::setw(14) << res.parse_par * 1e9 / n << std::setw(14)
		          << res.format_par * 1e9 / n << "\n";
	}
	for (size_t j = 0; j < prototypes.size(); j++) {
		const auto& res = proto_results[j];
		std::cout << std::left << std::setw(8) << benches[prototypes[j].first].label
		          << std::setw(13) << "libc_us" << std::right << std::fixed << std::setprecision(1)
		          << std::setw(14) << res.parse_seq * 1e9 / n << std::setw(14)
		          << res.parse_par * 1e9 / n << std::setw(14) << res.format_par * 1e9 / n << "\n";
	}
	return 0;
}

} // namespace

int leak()
{
	const case_t plain{"yyyy-MM-dd HH:mm:ss,SSS", "2016-01-01 12:00:00,123",
	                   expected_t{2016, 1, 1, 12, 0, 0, 123000, 0}, "", "", ""};
	const case_t zoned{"yyyy-MM-dd HH:mm:ss.SSS XXX", "2014-11-07 12:12:01.123 +05:30",
	                   expected_t{2014, 11, 7, 12, 12, 1, 123000, 330}, "", "", ""};
	check("first", "datetime_ms", plain.icu, plain);
	check("zoned", "datetime_ms", zoned.icu, zoned);
	check("again", "datetime_ms", plain.icu, plain);
	std::thread([&]() { check("thread", "datetime_ms", plain.icu, plain); }).join();
	return 0;
}

/**
 * Prints the formatter chosen for each time format read from the standard input
 */
int select()
{
	std::string tf;
	while (std::getline(std::cin, tf)) {
		const pvcop::formatter_desc fd = PVRush::PVFormat::get_datetime_formatter_desc(tf);
		std::cout << tf << "\t" << fd.name() << "\t" << fd.parameters() << "\n";
	}
	return 0;
}

int main(int argc, char** argv)
{
	if (argc >= 2 and std::string(argv[1]) == "select") {
		return select();
	}
	if (argc >= 2 and std::string(argv[1]) == "leak") {
		return leak();
	}
	if (argc >= 2 and std::string(argv[1]) == "bench") {
		const size_t n = argc >= 3 ? std::stoul(argv[2]) : 1000000;
		const int rounds = argc >= 4 ? std::stoi(argv[3]) : 5;
		return bench(n, rounds);
	}
	return probe();
}
