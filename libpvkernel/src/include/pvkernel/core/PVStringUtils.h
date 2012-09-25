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
	static inline uint64_t compute_str_factor16(uint16_t const* str, size_t size, bool case_sensitive = true)
	{
		// Using QString::toUtf8 and not QString::toLocal8Bit(), so that the "factor" variable
		// isn't system-locale dependant.
		/*QByteArray value_as_qba;
		if (case_sensitive) {
			value_as_qba = str.toUtf8();
		}
		else {
			value_as_qba = str.toLower().toUtf8();
		}

		const char* value_as_char_p = value_as_qba.data();*/
		const uint16_t* u16_buf = str;

		// AG: this is now historical, but still interesting !
		// This is a reduction. Do this with a for loop, so that we have more chance that the compiler
		// optimize this with vectorized operations.
		// Hmm.. spoken too fast... even with -ffast-math:
		// $ g++ -ffast-math -ftree-vectorizer-verbose=8 -march=native -O3 -I/home/aguinet/pv/libpicviz/src/include -I/home/aguinet/pv/libpvkernel/core/src/include -I/usr/include/qt4/QtCore -I/usr/include/qt4/ -I/home/aguinet/pv/libpvkernel/rush/src/include -I/usr/include/qt4/QtXml string_default.cpp 
		//
		// string_default.cpp:31: note: === vect_analyze_slp ===
		// string_default.cpp:31: note: === vect_make_slp_decision ===
		// string_default.cpp:31: note: === vect_detect_hybrid_slp ===
		// string_default.cpp:31: note: Vectorizing an unaligned access.
		// string_default.cpp:31: note: vect_model_load_cost: unaligned supported by hardware.
		// string_default.cpp:31: note: vect_model_load_cost: inside_cost = 2, outside_cost = 0 .
		// string_default.cpp:31: note: not vectorized: relevant stmt not supported: D.56753_14 = (float) D.56752_13;
		//
		// string_default.cpp:8: note: vectorized 0 loops in function.
		//
		// It looks like gcc can't handle the conversion from 8-bit char to 32-bit float when loading vector registers, because with 'factor' defined as an uint64_t :
		// $ g++ -fassociative-math -ffast-math -ftree-vectorizer-verbose=2 -O3 -I/home/aguinet/pv/libpicviz/src/include -I/home/aguinet/pv/libpvkernel/core/src/include -I/usr/include/qt4/QtCore -I/usr/include/qt4/ -I/home/aguinet/pv/libpvkernel/rush/src/include -I/usr/include/qt4/QtXml string_default.cpp 
		//
		// string_default.cpp:46: note: LOOP VECTORIZED.
		// string_default.cpp:8: note: vectorized 1 loops in function.
		//
		// So we define factor as an uint64_t, and convert it back to float after the reduction

		// With 128 the maximum ascii character value, the string must be greater than
		// (2**64-1)/128 characters (which is about 130 "tera-characters") before factor overflows...
		// I think we're pretty safe here !
		uint64_t factor_int = 0;
		if (case_sensitive) {
			for (size_t i = 0; i < size; i++) {
				factor_int += u16_buf[i];
			}
		}
		else {
			// TODO: hand-made vectorisation!
			for (size_t i = 0; i < size; i++) {
				factor_int += QChar::toLower(u16_buf[i]);
			}
		}

		return factor_int;
	};

	static inline uint64_t compute_str_factor(PVCore::PVUnicodeString const& str, bool /*case_sensitive*/)
	{
		const uint8_t* u8_buf = (const uint8_t*) str.buffer();
		size_t size = str.size();

		uint64_t factor_int = 0;
		for (size_t i = 0; i < size; i++) {
			factor_int += u8_buf[i];
		}

		return factor_int;
	};
};

}

#endif
