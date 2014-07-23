/**
 * \file PVUnicodeString.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVUNICODESTRING_H
#define PVCORE_PVUNICODESTRING_H

#include <pvbase/export.h>
#include <pvkernel/core/PVPythonClassDecl.h>

#include <stdlib.h>

namespace PVCore {

class PVBufferSlice;
class PVUnicodeString;
class PVUnicodeStringHashNoCase;

}

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVBufferSlice.h>

namespace PVCore {
/*! \brief Defines a read-only UTF8 string object.
 *  Design for performance, and does not use any explicit or implicit sharing as opposed to
 *  QString objects.
 *
 *  It is forced to be aligned on 4-byte (instead of 8-byte in 64 bits) for memory consumption issues.
 *  TODO: check the impact on performances !
 *
 *  These objects are constructed from PVBufferSlice.
 */
#pragma pack(push)
#pragma pack(4)
class LibKernelDecl PVUnicodeString
{
public:
	typedef uint8_t utf_char;
public:
	// Inline constructors
	PVUnicodeString(PVBufferSlice const& buf)
	{
		set_from_slice(buf);
	}

	PVUnicodeString()
	{
		_buf = NULL;
		_len = 0;
	}

	/*
	PVUnicodeString(PVUnicodeString const& other)
	{
		_buf = other._buf;
		_len = other._len;
	}*/

	PVUnicodeString(const utf_char* buf, size_t len):
		_buf(buf),
		_len(len)
	{
		assert(buf);
	}

	PVUnicodeString(const char* buf, size_t len):
		_buf((const utf_char*) buf),
		_len(len)
	{
		assert(buf);
	}
public:
	// == Conversions ==
	double to_double(bool& ok) const;
	float to_float(bool& ok) const;

	template <typename T>
	T to_integer() const
	{
		// TODO: implement this !
		T ret = 0;
		return ret;
	}

	// == Comparaisons ==
	// By default, memory-based comparaison are made
	bool operator==(const PVUnicodeString& o) const;
	bool operator!=(const PVUnicodeString& o) const;
	bool operator<(const PVUnicodeString& o) const;
	int compare(const PVUnicodeString& o) const;
	int compare(const char* str) const;
	int compareNoCase(const PVUnicodeString& o) const;

	// == Data access ==
	inline const utf_char* buffer() const { return _buf; }
	inline uint32_t size() const { return _len; };
	inline uint32_t len() const { return _len; };
	inline QString get_qstr() const
	{
		return QString::fromUtf8((const char*) _buf, _len);
	}
	inline QString get_qstr_copy() const
	{
		return get_qstr();
	}
	inline QString& get_qstr(QString& s) const
	{
		// We can't get away from that QString's allocation... :/
		s = QString::fromUtf8((const char*) _buf, _len);
		return s;
	}

	// == Data set ==
	inline void set_from_slice(PVBufferSlice const& buf)
	{
		_buf = (utf_char*) buf.begin();
		_len = buf.size()/sizeof(utf_char);
	}
	/*
	PVUnicodeString& operator=(const PVUnicodeString& other) 
	{
		_buf = other._buf;
		_len = other._len;
		_qstr.clear();
		return *this;
	}
	*/

	inline long to_long(int base) const
	{
		return strtol((const char*) _buf, nullptr, base);
	}

	inline unsigned long to_ulong(int base) const
	{
		return strtoul((const char*) _buf, nullptr, base);
	}

private:
	const utf_char* _buf;
	uint32_t _len;
	//mutable QString _qstr;

	PYTHON_EXPOSE()
};
#pragma pack(pop)

class PVUnicodeStringHashNoCase
{
public:
	PVUnicodeStringHashNoCase(PVUnicodeString const& str): _str(str) { }
public:
	inline PVUnicodeString const& str() const { return _str; }
	inline bool operator==(const PVUnicodeStringHashNoCase& o) const { return _str.compareNoCase(o.str()) == 0; }
private:
	PVUnicodeString const& _str;
};

}

#endif
