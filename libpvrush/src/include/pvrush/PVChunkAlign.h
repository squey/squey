#ifndef PVCHUNKALIGN_FILE_H
#define PVCHUNKALIGN_FILE_H

#include <pvcore/general.h>
#include <pvcore/PVChunk.h>

namespace PVRush {

// 
/*! \brief Interface class for chunk alignement.
 * A PVChunkAlign object is responsible for creating the elements of the chunk
 * and align the end of a chunk on an element's end.
 *
 * The default implementation does nothing but create an element that contains all the chunk's data.
 */
class LibRushDecl PVChunkAlign {
public:
	/*! \brief Alignement function
	 *  \param[in,out] cur_chunk Chunk whose data has just been read and that needs to be aligned
	 *  \param[in,out] next_chunk Chunk that can receive the data that should not be in cur_chunk (after alignement)
	 */
	virtual bool operator()(PVCore::PVChunk &cur_chunk, PVCore::PVChunk &next_chunk);
};

}

#endif
