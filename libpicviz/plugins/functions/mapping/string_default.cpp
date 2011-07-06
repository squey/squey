#include <stdio.h>

#ifdef WIN32
	#include <pvcore/win32-vs2008-stdint.h>
#else
	#include <stdint.h>
#endif

#include <picviz/general.h>
#include <picviz/PVMapping.h>

LibCPPExport float picviz_mapping_exec(const Picviz::PVMapping_p mapping, PVCol index, QString &value, void *userdata, bool is_first)
{

/* 16105 is the value corresponding to the arbitrary string:
 * "The competent programmer is fully aware of the limited size of his own skull. He therefore approaches his task with full humility, and 
 * avoids clever tricks like the plague."
 */
#define STRING_MAX_YVAL 16105

struct string_ascii_val_t {
	float max;
};

	uint64_t factor_int = 0;
	float retval;
	struct string_ascii_val_t string_ascii_val;

	// Using QString::toUtf8 and not QString::toLocal8Bit(), so that the "factor" variable
	// isn't system-locale dependant.
	QByteArray value_as_qba = value.toUtf8();
	const char* value_as_char_p = value_as_qba.data();
	int size = value_as_qba.size();

	// This is a reduction. Do this with a for loop, so that we have more chance that the compiler
	// optimize this with vectorized operations.
	// Hmm.. spoken too fast... even with -ffast-math:
	// $ g++ -ffast-math -ftree-vectorizer-verbose=8 -march=native -O3 -I/home/aguinet/pv/libpicviz/src/include -I/home/aguinet/pv/libpvcore/src/include -I/usr/include/qt4/QtCore -I/usr/include/qt4/ -I/home/aguinet/pv/libpvrush/src/include -I/usr/include/qt4/QtXml string_default.cpp 
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
	// $ g++ -fassociative-math -ffast-math -ftree-vectorizer-verbose=2 -O3 -I/home/aguinet/pv/libpicviz/src/include -I/home/aguinet/pv/libpvcore/src/include -I/usr/include/qt4/QtCore -I/usr/include/qt4/ -I/home/aguinet/pv/libpvrush/src/include -I/usr/include/qt4/QtXml string_default.cpp 
	//
	// string_default.cpp:46: note: LOOP VECTORIZED.
	// string_default.cpp:8: note: vectorized 1 loops in function.
	//
	// So we define factor as an uint64_t, and convert it back to float after the reduction
	
	for (int i = 0; i < size; i++) {
		// With 128 the maximum ascii character value, the string must be greater than
		// (2**64-1)/128 characters (which is about 130 "tera-characters") before factor overflows...
		// I think we're pretty safe here !
		factor_int += value_as_char_p[i];
	}

	float factor = (float) factor_int;

	if ( is_first ) {
		string_ascii_val.max = factor;
		PICVIZ_USERDATA(userdata, struct string_ascii_val_t) = string_ascii_val;
	} else {
		string_ascii_val = PICVIZ_USERDATA(userdata, struct string_ascii_val_t);
		if ( factor > string_ascii_val.max ) {
			string_ascii_val.max = factor;
			PICVIZ_USERDATA(userdata, struct string_ascii_val_t) = string_ascii_val;
		}
	}

	if ( string_ascii_val.max > STRING_MAX_YVAL ) {
		retval = factor / string_ascii_val.max;
	} else {
		retval = factor / STRING_MAX_YVAL;
	}

	return retval;
}

LibCPPExport int picviz_mapping_init()
{
	return 0;
}

LibCPPExport int picviz_mapping_terminate()
{
	return 0;
}

LibCPPExport int picviz_mapping_test()
{
	return 0;
}
