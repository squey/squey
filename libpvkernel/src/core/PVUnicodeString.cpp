/**
 * \file PVUnicodeString.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVPython.h>
#include <pvkernel/core/PVPythonClassDecl.h>
#include <pvkernel/core/PVUnicodeString.h>

#include <unicode/ustring.h>
#include <unicode/coll.h>

#include <tbb/enumerable_thread_specific.h>

// Taken from QT 4.7.3's source code
// The original disclamer is:
/*
 *  These functions are based on Peter J. Weinberger's hash function
 *  (from the Dragon Book). The constant 24 in the original function
 *  was replaced with 23 to produce fewer collisions on input such as
 *  "a", "aa", "aaa", "aaaa", ...
 */
static inline unsigned int hash(const PVCore::PVUnicodeString::utf_char *p, int n)
{
	unsigned int h = 0; 

	// TODO: see if GCC vectorize this
	while (n--) {
		h = (h << 4) + (*p++);
		h ^= (h & 0xf0000000) >> 23;
		h &= 0x0fffffff;
	}
	return h;
}

static inline unsigned int hash_lowercase(const PVCore::PVUnicodeString::utf_char *p, int n)
{
	return hash(p, n);
}

unsigned int qHash(PVCore::PVUnicodeString const& str)
{
	return hash(str.buffer(), str.size());
}

unsigned int qHash(PVCore::PVUnicodeStringHashNoCase const& str)
{
	return hash_lowercase(str.str().buffer(), str.str().size());
}

double PVCore::PVUnicodeString::to_double(bool& /*ok*/) const
{
	return 0.0;
}

float PVCore::PVUnicodeString::to_float(bool& /*ok*/) const
{
	return 0.0;
}

bool PVCore::PVUnicodeString::operator==(const PVUnicodeString& o) const
{
	if (_len != o._len) {
		return false;
	}
	return memcmp(_buf, o._buf, _len*sizeof(utf_char)) == 0;
}

bool PVCore::PVUnicodeString::operator!=(const PVUnicodeString& o) const
{
	if (_len != o._len) {
		return true;
	}
	return memcmp(_buf, o._buf, _len*sizeof(utf_char)) != 0;
}

bool PVCore::PVUnicodeString::operator<(const PVUnicodeString& o) const
{
	const utf_char* a = _buf;
	const utf_char* b = o._buf;
	int ret = memcmp(a, b, picviz_min(_len, o._len)*sizeof(utf_char));
	if (ret == 0) {
		return _len < o._len;
	}
	return false;

}

int PVCore::PVUnicodeString::compare(const char* str) const
{
	size_t str_size = strlen(str);
	return compare(PVUnicodeString((PVCore::PVUnicodeString::utf_char*) str, str_size));
}

int PVCore::PVUnicodeString::compare(const PVUnicodeString& o) const
{
	int ret = memcmp(_buf, o._buf, picviz_min(o._len, _len)*sizeof(utf_char));
	if (ret == 0) {
		if (_len < o._len) {
			ret = -1;
		}
		else
		if (_len > o._len) {
			ret = 1;
		}
	}
	return ret;
}

int PVCore::PVUnicodeString::compareNoCase(const PVUnicodeString& o) const
{
	static tbb::enumerable_thread_specific<Collator*> tls_collator(
			[=]
			{
				UErrorCode err = U_ZERO_ERROR;
				Collator* c = Collator::createInstance(err);
				c->setStrength(Collator::PRIMARY);
				return c;
			});
	Collator* const c = tls_collator.local();
	UErrorCode err = U_ZERO_ERROR;
	const UCollationResult res = c->compareUTF8(StringPiece((const char*) _buf, _len), StringPiece((const char*) o._buf, o._len), err);
	return (int) res;
}

PYTHON_EXPOSE_IMPL(PVCore::PVUnicodeString)
{
	int (PVUnicodeString::*fp_compare_const_str)(const char*) const;
	int (PVUnicodeString::*fp_compare_obj)(PVCore::PVUnicodeString const&) const;
	fp_compare_const_str = &PVUnicodeString::compare;
	fp_compare_obj = &PVUnicodeString::compare;
	boost::python::class_<PVCore::PVUnicodeString>("PVUnicodeString")
		.def("len", &PVUnicodeString::len)
		.def("size", &PVUnicodeString::size)
		.def("compare", fp_compare_const_str)
		.def("compare_alias", fp_compare_obj)
	;
}
