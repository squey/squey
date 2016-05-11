#include <pvkernel/core/inendi_assert.h>
#include <pvguiqt/PVSortFilter.h>

#include <random>
#include <chrono>
#include <iostream>

#ifdef INSPECTOR_BENCH
constexpr size_t SIZE = 200000000;
#else
constexpr size_t SIZE = 10000;
#endif

int main()
{
	PVGuiQt::PVSortFilter sf(SIZE);
	Inendi::PVSelection sel(SIZE);
	sel.select_even();

	PV_VALID(sf.size(), SIZE);

	// Shuffle values as it is the worse but normal case)
	auto& sort = sf.sorting().to_core_array();
	std::mt19937 g; // Init from seed to keep same random ordering
	std::shuffle(sort.begin(), sort.end(), g);

	auto start = std::chrono::steady_clock::now();
	sf.set_filter(sel);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff(end - start);

	PV_VALID(sf.size(), SIZE / 2);

#ifndef INSPECTOR_BENCH
	PV_VALID(sf.row_pos_to_index(5), (PVRow)7157);
#endif

	start = std::chrono::steady_clock::now();
	sf.set_filter_as_sort();
	end = std::chrono::steady_clock::now();
	diff += (end - start);

#ifndef INSPECTOR_BENCH
	PV_VALID(sf.row_pos_to_index(10), (PVRow)7157);
#endif

	std::cout << diff.count();
}
