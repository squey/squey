/**
 * \file PVStringUtils.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_STRINGUTILS_H
#define PVCORE_STRINGUTILS_H

/* 16105 is the value corresponding to the arbitrary string:
 * "The competent programmer is fully aware of the limited size of his own skull. He therefore approaches his task with full humility, and 
 * avoids clever tricks like the plague."
 * Used by compte_str_factor
 */
#define STRING_MAX_YVAL 16105

#include <QChar>

namespace PVCore {

class PVStringUtils {
public:

	static inline uint32_t compute_str_factor16(uint16_t const* str, size_t size, bool case_sensitive = true)
	{
		return _compute_str_factor<uint16_t, 9>(str, size, case_sensitive);
	}

	static inline uint32_t compute_str_factor(PVCore::PVUnicodeString const& str, bool /*case_sensitive*/)
	{
		const uint8_t* u8_buf = (const uint8_t*) str.buffer();
		size_t size = str.size();

		return _compute_str_factor<uint8_t, 8>(u8_buf, size, false);
	};

private:
	template <typename T, size_t char_bits>
	static inline uint32_t _compute_str_factor(T const* buf, size_t size, bool case_sensitive = true)
	{
		if (size < 1) {
			return 0;
		}

		// -------------------------------------------------------------------
		// |  a (4)  |  b (0..12)  |      c (9)        |      d (7..19)      | 32 bits
		// -------------------------------------------------------------------
		// a : log2(length)
		// b : linear splitting between 'a' according to the length
		// c : sum of the first 2 bytes
		// d : weak bits of the sum of the remaining bytes

		size_t max_size = std::min(size, 4096UL);

		size_t a = log2(max_size);
		size_t a_prime = std::min(a, 9UL);
		uint32_t factor = (a+1) << (32-4); // +1 to separate 1 length strings from 0 length strings

		size_t b = (max_size - (1 << a));
		factor = factor | (b << (32-4-a));

		const T& c = buf[0];
		size_t d_bits = 32-4-a-char_bits;
		factor = factor | (char_value(c) << d_bits);

		size_t d = 0;
		size_t max_remaining_size = std::min(size, (size_t) 1 << (32-4-a_prime-char_bits-char_bits)) -1;
		if (case_sensitive) {
			for (size_t i = 0; i < max_remaining_size; i++) {
				const T& c = buf[i+1];
				d += char_value(c);
			}
		}
		else {
			for (size_t i = 0; i < max_remaining_size; i++) {
				T c = to_lower(buf[i+1]);
				d += char_value(c);
			}
		}
		size_t shift = std::max(0UL, (d_bits) - (size_t) ceil(log2(max_remaining_size * (1 << char_bits))));
		factor = factor | ((d & ((1 << d_bits)-1)) << shift);

		return factor;
	}

	static inline uint32_t char_value(uint16_t c)
	{
		return (c >> 8) + (c & 0x00FF);
	}

	static inline uint32_t char_value(uint8_t c)
	{
		return c;
	}

	static inline uint16_t to_lower(uint16_t c)
	{
		return QChar::toLower(c);
	}

	static inline uint16_t to_lower(uint8_t c)
	{
		return c;
	}
};

}

#endif
