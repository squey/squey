/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#ifndef INENDI_PVCOMBCOL_H
#define INENDI_PVCOMBCOL_H

#include <cstdint>
#include <limits>

#define NO_HARD_CHECK 1

namespace Inendi
{

struct PVCombCol {
#ifdef NO_HARD_CHECK
	constexpr PVCombCol(int32_t v) : value(v) {}

	operator int32_t() { return value; }
#else
	constexpr explicit PVCombCol(int32_t v) : value(v) {}
#endif

	bool operator<(PVCombCol c) const { return value < c.value; }
	bool operator==(PVCombCol c) const { return value == c.value; }

	int32_t value;
};

constexpr static PVCombCol INVALID_COMB_COL(std::numeric_limits<int32_t>::max());
}

#endif /* INENDI_PVAXESCOMBINATION_H */
