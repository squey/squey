/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#ifndef INENDI_PVCOMBCOL_H
#define INENDI_PVCOMBCOL_H

#include <cstddef>

#define NO_HARD_CHECK 1

namespace Inendi
{

struct PVCombCol {
#ifdef NO_HARD_CHECK
	constexpr PVCombCol(size_t v) : value(v) {}

	operator long unsigned() { return value; }
#else
	constexpr explicit PVCombCol(size_t v) : value(v) {}
#endif

	bool operator<(PVCombCol c) const { return value < c.value; }
	bool operator==(PVCombCol c) const { return value == c.value; }

	size_t value;
};

constexpr static PVCombCol INVALID_COMB_COL(-1);
}

#endif /* INENDI_PVAXESCOMBINATION_H */
