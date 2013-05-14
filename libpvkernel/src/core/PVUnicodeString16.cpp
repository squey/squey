/**
 * \file PVUnicodeString16.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVPython.h>
#include <pvkernel/core/PVPythonClassDecl.h>
#include <pvkernel/core/PVUnicodeString16.h>

#include <unicode/ustring.h>

// Taken from QT 4.7.3's source code
// The original disclamer is:
/*
 *  These functions are based on Peter J. Weinberger's hash function
 *  (from the Dragon Book). The constant 24 in the original function
 *  was replaced with 23 to produce fewer collisions on input such as
 *  "a", "aa", "aaa", "aaaa", ...
 */
static inline unsigned int hash(const PVCore::PVUnicodeString16::utf_char *p, int n)
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

static inline unsigned int hash_lowercase(const PVCore::PVUnicodeString16::utf_char *p, int n)
{
	unsigned int h = 0; 
	const QChar* qp = (const QChar*) p;
	const QChar* qe = ((const QChar*) p) + n;

	// Inspired by QString::toLower code source !
	// Avoid one check in the following loop !
	if (qp->isLowSurrogate()) {
		qp++;
	}

	// TODO: some work can be done for vectorizing this process !
	while (qp < qe) {
		uint32_t lc;
		if (qp->isLowSurrogate() && (qp-1)->isHighSurrogate()) {
			lc = QChar::toLower(QChar::surrogateToUcs4(*(qp-1), *qp));
		}
		else {
			lc = qp->toLower().unicode();
		}
		h = (h << 4) + lc;
		h ^= (h & 0xf0000000) >> 23;
		h &= 0x0fffffff;
		qp++;
	}
	return h;
}

unsigned int qHash(PVCore::PVUnicodeString16 const& str)
{
	return hash(str.buffer(), str.size());
}

unsigned int qHash(PVCore::PVUnicodeString16HashNoCase const& str)
{
	return hash_lowercase(str.str().buffer(), str.str().size());
}

double PVCore::PVUnicodeString16::to_double(bool& /*ok*/) const
{
	return 0.0;
}

float PVCore::PVUnicodeString16::to_float(bool& /*ok*/) const
{
	return 0.0;
}

bool PVCore::PVUnicodeString16::operator==(const PVUnicodeString16& o) const
{
	if (_len != o._len) {
		return false;
	}
	return memcmp(_buf, o._buf, _len*sizeof(utf_char)) == 0;
}

bool PVCore::PVUnicodeString16::operator!=(const PVUnicodeString16& o) const
{
	if (_len != o._len) {
		return true;
	}
	return memcmp(_buf, o._buf, _len*sizeof(utf_char)) != 0;
}

bool PVCore::PVUnicodeString16::operator<(const PVUnicodeString16& o) const
{
	const utf_char* a = _buf;
	const utf_char* b = o._buf;
	int ret = memcmp(a, b, picviz_min(_len, o._len)*sizeof(utf_char));
	if (ret == 0) {
		return _len < o._len;
	}
	return false;

}

int PVCore::PVUnicodeString16::compare(const char* str) const
{
	QString str_(str);
	const uint32_t str_size = str_.size();
	int ret = memcmp(str_.constData(), _buf, picviz_min(_len, str_size)*sizeof(utf_char));
	if (ret == 0) {
		if (_len < str_size) {
			ret = -1;
		}
		else
		if (_len > str_size) {
			ret = 1;
		}
	}
	return ret;
}

int PVCore::PVUnicodeString16::compare(const PVUnicodeString16& o) const
{
	int ret = memcmp(o._buf, _buf, picviz_min(o._len, _len)*sizeof(utf_char));
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

int PVCore::PVUnicodeString16::compareNoCase(const PVUnicodeString16& o) const
{
	UErrorCode err = U_ZERO_ERROR;
	return u_strCaseCompare((const UChar*) _buf, _len, (const UChar*) o._buf, o._len, 0, &err);
}

PYTHON_EXPOSE_IMPL(PVCore::PVUnicodeString16)
{
	int (PVUnicodeString16::*fp_compare_const_str)(const char*) const;
	int (PVUnicodeString16::*fp_compare_obj)(PVCore::PVUnicodeString16 const&) const;
	fp_compare_const_str = &PVUnicodeString16::compare;
	fp_compare_obj = &PVUnicodeString16::compare;
	boost::python::class_<PVCore::PVUnicodeString16>("PVUnicodeString16")
		.def("len", &PVUnicodeString16::len)
		.def("size", &PVUnicodeString16::size)
		.def("compare", fp_compare_const_str)
		.def("compare_alias", fp_compare_obj)
	;
}
