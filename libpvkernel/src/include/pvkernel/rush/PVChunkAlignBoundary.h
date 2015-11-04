/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCHUNKALIGNBOUNDARY_FILE_H
#define PVCHUNKALIGNBOUNDARY_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/rush/PVChunkAlign.h>

#include <QTextBoundaryFinder>

namespace PVRush {

/*! \brief Alignement class on a given boundary.
 * This class herits from PVChunkAlign and is used to align chunks on a boundary (cf. QTextBoundaryFinder)
 */
class PVChunkAlignBoundary : public PVChunkAlign {
public:
	PVChunkAlignBoundary(QTextBoundaryFinder::BoundaryType boundary);
public:
	virtual bool operator()(PVCore::PVChunk &cur_chunk, PVCore::PVChunk &next_chunk);
protected:
	QTextBoundaryFinder::BoundaryType _b;
};

}

#endif
