/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVRAWSOURCEBASE_FILE_H
#define PVRAWSOURCEBASE_FILE_H

#include <pvkernel/rush/PVRawSourceBase_types.h> // for PVRawSourceBase_p
#include <pvkernel/filter/PVFilterFunction.h> // for PVFilterFunctionBase
#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/core/PVBinaryChunk.h>
#include <stddef.h>
#include <QString> // for QString

#include "pvbase/types.h" // for PVCol, chunk_index

namespace PVCore
{
class PVChunk;
} // namespace PVCore

namespace PVRush
{

class PVRawSourceBase : public PVFilter::PVFilterFunctionBase<PVCore::PVChunk*, void>
{
  public:
	typedef PVRawSourceBase_p p_type;

  public:
	PVRawSourceBase();
	~PVRawSourceBase() override = default;
	PVRawSourceBase(const PVRawSourceBase& src) = delete;

  public:
	virtual void release_input(bool /*cancel_first*/ = false) {}

  public:
	chunk_index last_elt_index() { return _last_elt_index; }
	void set_number_cols_to_reserve(PVCol col)
	{
		if (col == 0) {
			col = PVCol(1);
		}
		_ncols_to_reserve = col;
	}
	PVCol get_number_cols_to_reserve() const { return _ncols_to_reserve; }
	virtual QString human_name() = 0;
	virtual void seek_begin() = 0;
	virtual void prepare_for_nelts(chunk_index nelts) = 0;
	virtual size_t get_size() const = 0;
	virtual PVCore::PVChunk* operator()() = 0;
	virtual EChunkType chunk_type() const = 0;

  protected:
	mutable chunk_index _last_elt_index; // Local file index of the last element of that source. Can
	                                     // correspond to a number of lines
	PVCol _ncols_to_reserve;
};

template <typename ChunkType>
class PVRawSourceBaseType : public PVRawSourceBase
{
  protected:
	using chunk_type_t = ChunkType;
	static constexpr auto chunk_type_v = ChunkType::chunk_type;

  public:
	using PVRawSourceBase::PVRawSourceBase;

  public:
	virtual EChunkType chunk_type() const { return chunk_type_v; }
};

} // namespace PVRush

#endif // PVRAWSOURCEBASE_FILE_H
