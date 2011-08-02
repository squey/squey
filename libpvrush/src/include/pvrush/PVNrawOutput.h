#ifndef PVNRAWOUTPUT_FILE_H
#define PVNRAWOUTPUT_FILE_H

#include <pvcore/general.h>
#include <pvbase/types.h>
#include <pvcore/PVChunk.h>
#include <pvrush/PVOutput.h>
#include <pvrush/PVNraw.h>

namespace PVRush {

class LibRushDecl PVNrawOutput : public PVRush::PVOutput {
public:
	typedef std::map<PVRow,PVCore::chunk_index> map_pvrow;
public:
	PVNrawOutput(PVNraw &nraw_dest);
public:
	// This is the output of a TBB pipeline
	// It takes a PVCore::PVChunk* as a parameter, and do whatever he wants with it
	// It *must* call PVChunk->free() in the end !!
	virtual void operator()(PVCore::PVChunk* out);
	map_pvrow const& get_pvrow_index_map() const;
	void clear_pvrow_index_map();
protected:
	PVNraw &_nraw_dest;
	map_pvrow _pvrow_chunk_idx;
};

}

#endif
