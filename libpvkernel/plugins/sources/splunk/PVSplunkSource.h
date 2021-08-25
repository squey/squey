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

#ifndef PVSPLUNKSOURCE_FILE_H
#define PVSPLUNKSOURCE_FILE_H

#include <pvkernel/core/PVTextChunk.h>

#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <QString>

#include "../../common/splunk/PVSplunkAPI.h"

namespace PVRush
{

class PVSplunkQuery;

class PVSplunkSource : public PVRawSourceBaseType<PVCore::PVTextChunk>
{
  public:
	PVSplunkSource(PVInputDescription_p input);

	virtual ~PVSplunkSource();

  public:
	QString human_name() override;
	void seek_begin() override;
	void prepare_for_nelts(chunk_index nelts) override;

	/**
	 * It should be the size of the Splunk source.
	 *
	 * As we don't have "free" metric, we return 0 for "unknown"
	 */
	size_t get_size() const override { return 0; }
	PVCore::PVTextChunk* operator()() override;

  protected:
	chunk_index _next_index;

  private:
	PVSplunkQuery& _query;
	PVSplunkAPI _splunk;
	bool _query_end = false;
};

} // namespace PVRush

#endif
