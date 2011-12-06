#ifndef PVCORE_PVUNICODESTRING_H
#define PVCORE_PVUNICODESTRING_H

#include <pvbase/export.h>

namespace PVCore {

class PVBufferSlice;
class PVUnicodeString;

}

#ifdef QHASH_H
#warning libpvkernel/core/PVUnicodeString.h must be included before QHash if you want to use it as a QHash key.
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
		_buf = (utf_char*) buf.begin();
		_len = buf.size()/sizeof(utf_char);
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
	bool compare(const PVUnicodeString& o) const;

	// == Data access ==
	inline const utf_char* buffer() const { return _buf; }
	inline size_t size() const { return _len; };
	inline size_t len() const { return _len; };

protected:
	const utf_char* _buf;
	size_t _len;
};

}

#endif
