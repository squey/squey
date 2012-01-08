#ifndef PVCORE_PVUNICODESTRING_H
#define PVCORE_PVUNICODESTRING_H

#include <pvbase/export.h>
#include <pvkernel/core/PVPythonClassDecl.h>

namespace PVCore {

class PVBufferSlice;
class PVUnicodeString;

}

#ifdef QHASH_H
//#warning libpvkernel/core/PVUnicodeString.h must be included before QHash if you want to use it as a QHash key.
#endif
LibKernelDecl unsigned int qHash(PVCore::PVUnicodeString const& str);

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVBufferSlice.h>

namespace PVCore {
/*! \brief Defines a read-only UTF16 string object.
 *  Design for performance, and does not use any explicit or implicit sharing as opposed to
 *  QString objects.
 *
 *  This objects are constructed from PVBufferSlice.
 */

class LibKernelDecl PVUnicodeString
{
public:
	typedef uint16_t utf_char;
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

	PVUnicodeString(PVUnicodeString const& other)
	{
		_buf = other._buf;
		_len = other._len;
	}

	PVUnicodeString(const utf_char* buf, size_t len):
		_buf(buf),
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
		T ret = 0;
	}

	// == Comparaisons ==
	// By default, memory-based comparaison are made
	bool operator==(const PVUnicodeString& o) const;
	bool operator!=(const PVUnicodeString& o) const;
	bool operator<(const PVUnicodeString& o) const;
	int compare(const PVUnicodeString& o) const;
	int compare(const char* str) const;

	// == Data access ==
	inline const utf_char* buffer() const { return _buf; }
	inline uint32_t size() const { return _len; };
	inline uint32_t len() const { return _len; };
	inline QString const& get_qstr() const
	{
		if (_qstr.isNull()) {
			_qstr.setRawData((QChar*) _buf, _len);
		}
		return _qstr;
	}

	// == Data set ==
	inline void set_from_slice(PVBufferSlice const& buf)
	{
		_buf = (utf_char*) buf.begin();
		_len = buf.size()/sizeof(utf_char);
	}
	PVUnicodeString& operator=(const PVUnicodeString& other) 
	{
		_buf = other._buf;
		_len = other._len;
		_qstr.clear();
		return *this;
	}

protected:
	const utf_char* _buf;
	uint32_t _len;
	mutable QString _qstr;

	PYTHON_EXPOSE()
};

}

#endif
