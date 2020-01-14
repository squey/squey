/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/serialize_numbers.h>

#include <pvlogger.h>

std::list<std::pair<size_t, size_t>> PVCore::deserialize_numbers(const std::string& numbers_list)
{
	std::list<std::pair<size_t, size_t>> ranges;

	size_t initial = 0; // Value of first number in a range.
	size_t value = 0;   // If nonzero, a number being accumulated.
	bool lhs_specified = false;
	bool rhs_specified = false;
	bool dash_found = false; // True if a '-' is found in this field.

	bool in_digits = false;

	// Special case: '--field=-' means all fields, emulate '--field=1-' .
	size_t pos = 0;
	if (numbers_list.compare("-") == 0) {
		value = 0;
		lhs_specified = true;
		dash_found = true;
		pos++;
	}

	while (true) {
		if (numbers_list[pos] == '-') {
			in_digits = false;
			// Starting a range.
			if (dash_found) {
				throw std::runtime_error("invalid range");
			}

			dash_found = true;
			pos++;

			initial = (lhs_specified ? value : 0);
			value = 0;
		} else if (numbers_list[pos] == ',' or std::isblank(numbers_list[pos]) or
		           numbers_list[pos] == '\0') {
			in_digits = false;
			// Ending the string, or this field/byte sublist.
			if (dash_found) {
				dash_found = false;

				if (not lhs_specified and not rhs_specified) {
					// if a lone dash is allowed, emulate '1-' for all fields
					initial = 0;
				}

				// A range.  Possibilities: -n, m-n, n-.
				// In any case, 'initial' contains the start of the range.
				if (not rhs_specified) {
					// 'n-'.  From 'initial' to end of line.
					ranges.emplace_back(initial, UINTMAX_MAX);
				} else {
					// 'm-n' or '-n' (1-n).
					if (value < initial) {
						throw std::runtime_error("invalid decreasing range");
					}

					ranges.emplace_back(initial, value);
				}
				value = 0;
			} else {

				ranges.emplace_back(value, value);
				value = 0;
			}

			if (numbers_list[pos] == '\0') {
				break;
			}

			pos++;
			lhs_specified = false;
			rhs_specified = false;
		} else if (std::isdigit(numbers_list[pos])) {
			in_digits = true;

			if (dash_found) {
				rhs_specified = true;
			} else {
				lhs_specified = true;
			}

			DECIMAL_DIGIT_ACCUMULATE(value, numbers_list[pos] - '0', uintmax_t);

			pos++;
		} else {
			throw std::runtime_error("invalid field value");
		}
	}

	if (ranges.size() == 0) {
		throw std::runtime_error("missing list of fields");
	}

	ranges.sort();

	// Merge range pairs (e.g. `2-5,3-4' becomes `2-5').
#if 0
	for (auto i = ranges.begin(); i != ranges.end(); i++) {
		for (auto j = std::next(ranges.begin()); j != ranges.end(); j++) {
			if (j->first <= i->second) {
				i->second = std::max(j->second, i->second);
				ranges.erase(j); // FIXME
				j--;
			} else {
                break;
            }
		}
	}
#endif

	return ranges;
}

size_t PVCore::get_count_from_ranges(const std::list<std::pair<size_t, size_t>>& ranges)
{
	size_t count = 0;

	for (auto [begin, end] : ranges) {
		count += end - begin + 1;
	}

	return count;
}