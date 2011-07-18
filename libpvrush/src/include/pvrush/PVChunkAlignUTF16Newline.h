#ifndef PVCHUNKALIGNUTF16NEWLINE_FILE_H
#define PVCHUNKALIGNUTF16NEWLINE_FILE_H

#include <pvcore/general.h>
#include <pvcore/PVChunk.h>
#include <pvrush/PVChunkAlign.h>
#include <pvrush/PVChunkAlignUTF16Char.h>

#include <QString>

namespace PVRush {

/*! \brief Alignement class on an UTF16 characters.
 * This class herits from PVChunkAlign and is used to align chunks on an UTF16 characters.
 *
 * \todo Currently, this class is used to align text file chunks on '\\n'.
 * It would be better to have a PVChunkAlignBoundary that uses QTextBoundaryFinder.
 */
class LibRushDecl PVChunkAlignUTF16Newline : public PVChunkAlign {
public:
	PVChunkAlignUTF16Newline();
public:
	virtual bool operator()(PVCore::PVChunk &cur_chunk, PVCore::PVChunk &next_chunk);
protected:
	PVChunkAlignUTF16Char _align_char;
};

}

#endif
