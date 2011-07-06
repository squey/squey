#ifndef PVCHUNKTRANSFORM_FILE_H
#define PVCHUNKTRANSFORM_FILE_H

#include <pvcore/general.h>

#ifdef WIN32
	#include <pvcore/win32-vs2008-stdint.h>
#else
	#include <stdint.h>
#endif

namespace PVRush {

/*! \brief These filters transform chunk just after being read from the source.
 * This is used (for now) in order to convert text inputs to UTF16
 *
 * \sa PVChunkTransformUTF16
 */
class LibExport PVChunkTransform
{
public:
	/*! \brief Returns the number of bytes that should be read for the following chunk, knowning the original size.
	 * This allow the first chunk allocation to allocate enough memory for the transformed chunk, if that can be
	 * predicted.
	 */
	virtual size_t next_read_size(size_t org_size) const;

	/*! \brief Transformation interface
	 *  \param[in] data Pointer to a chunk's data
	 *  \param[in] len_read Number of bytes in data that have been read from the original source
	 *  \param[in] len_avail Total number of available bytes in data. Note that len_avail includes len_read (len_avail > len_read)
	 *  \return The final size of the data. It must be less than len_avail.
	 */
	virtual size_t operator()(char* data, size_t len_read, size_t len_avail) const;
};

}

#endif
