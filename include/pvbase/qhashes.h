#ifndef PVBASE_QHASHES_H
#define PVBASE_QHASHES_H

// Defines qhash overrides
// They need to be included before QHash !
#ifdef QHASH_H
#error pvbase/qhashes.h must be included before QHash !
#endif

#include <pvkernel/core/string_tbb.h>

// Forward declarations
namespace PVCore {
class PVUnicodeString;
class PVUnicodeStringHashNoCase;
class PVUnicodeString16;
class PVUnicodeString16HashNoCase;
}

unsigned int qHash(PVCore::PVUnicodeString const& str);
unsigned int qHash(PVCore::PVUnicodeStringHashNoCase const& str);

unsigned int qHash(PVCore::PVUnicodeString16 const& str);
unsigned int qHash(PVCore::PVUnicodeString16HashNoCase const& str);

unsigned int qHash(std::string_tbb const& str);

#endif
