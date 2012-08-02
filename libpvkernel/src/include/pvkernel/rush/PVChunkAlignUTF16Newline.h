/**
 * \file PVChunkAlignUTF16Newline.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCHUNKALIGNUTF16NEWLINE_FILE_H
#define PVCHUNKALIGNUTF16NEWLINE_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/rush/PVChunkAlign.h>
#include <pvkernel/rush/PVChunkAlignUTF16Char.h>

#include <QString>

namespace PVRush {

/*! \brief Alignement class on an UTF16 characters.
 * This class herits from PVChunkAlign and is used to align chunks on an UTF16 characters.
 *
 * \todo Currently, this class is used to align text file chunks on '\\n'.
 * It would be better to have a PVChunkAlignBoundary that uses QTextBoundaryFinder.
 */
class LibKernelDecl PVChunkAlignUTF16Newline : public PVChunkAlign {
public:
	PVChunkAlignUTF16Newline();
public:
	virtual bool operator()(PVCore::PVChunk &cur_chunk, PVCore::PVChunk &next_chunk);
protected:
	PVChunkAlignUTF16Char _align_char;
};

}

#endif
