#ifndef PVCHUNKALIGNHADOOP_FILE_H
#define PVCHUNKALIGNHADOOP_FILE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/rush/PVChunkAlign.h>

#include <QString>

namespace PVRush {

class PVInputHadoop;

class PVChunkAlignHadoop: public PVChunkAlign {
	friend class PVInputHadoop;
	enum {
		MAX_ELEMENT_LENGTH = 1024*1024
	};
public:
	typedef uint64_t offset_type;
	typedef uint32_t element_length_type;
	typedef uint32_t field_length_type;
protected:
	PVChunkAlignHadoop(PVInputHadoop& input);
public:
	virtual bool operator()(PVCore::PVChunk &cur_chunk, PVCore::PVChunk &next_chunk);
protected:
	// Used to set the last offset seen (for PVInputHadoop::current_offset)
	PVInputHadoop& _input;
};

}

#endif
