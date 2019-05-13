/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#ifndef __PVERFSOURCE_FILE_H__
#define __PVERFSOURCE_FILE_H__

#include <iterator>
#include <fcntl.h>
#include <memory>

#include <QString>

#include <pvkernel/core/PVChunk.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include "../../common/erf/PVERFDescription.h"
#include "PVERFSource.h"

namespace PVRush
{

template <template <class T> class Allocator = PVCore::PVMMapAllocator>
class PVERFSource : public PVRawSourceBase
{
  public:
	PVERFSource(PVInput_p input, PVERFDescription* input_desc, size_t chunk_size) {}

	QString human_name() override { return ""; };              // FIXME
	void seek_begin() override{};                              // FIXME
	void prepare_for_nelts(chunk_index nelts) override{};      // FIXME
	size_t get_size() const override { return 0; };            // FIXME
	PVCore::PVChunk* operator()() override { return nullptr; } // FIXME
};

} // namespace PVRush

#endif // __PVERFSOURCE_FILE_H__
