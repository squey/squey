#include <pvkernel/core/PVUnicodeString.h>


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

unsigned int qHash(PVCore::PVUnicodeString const& str)
{
	return hash(str.buffer(), str.size());
}

double PVCore::PVUnicodeString::to_double(bool& ok) const
{
	return 0.0;
}

float PVCore::PVUnicodeString::to_float(bool& ok) const
{
	return 0.0;
}

bool PVCore::PVUnicodeString::operator==(const PVUnicodeString& o) const
{
	// TODO: vectorize this by hand and compare
	return memcmp(_buf, o._buf, picviz_min(_len, o._len)) == 0;
}

bool PVCore::PVUnicodeString::operator!=(const PVUnicodeString& o) const
{
	return memcmp(_buf, o._buf, picviz_min(_len, o._len)) != 0;
}

bool PVCore::PVUnicodeString::compare(const PVUnicodeString& o) const
{
	return memcmp(_buf, o._buf, picviz_min(_len, o._len)) < 0;
}
