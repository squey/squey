/* * MIT License
 *
 * © Squey, 2026
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

/**
 * Check that cancelling an import that failed leaves its failure to wait_end().
 *
 * An import is cancelled where nothing would catch that failure: from the handler of a thread
 * interruption, or once the user chose to stop an import running out of memory.
 */

#include <pvkernel/core/squey_assert.h>

#include "common.h"

#include <chrono>
#include <stdexcept>
#include <thread>

const std::string filename = TEST_FOLDER "/picviz/heat_line.csv";
const std::string fileformat = TEST_FOLDER "/picviz/heat_line.csv.format";

namespace
{

struct output_failure : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

/**
 * An output failing on the first chunk it is given, as the NRaw does once its files can no
 * longer grow.
 */
class PVFailingOutput : public PVRush::PVOutput
{
  public:
	void operator()(PVCore::PVChunk* out) override
	{
		out->free();
		throw output_failure("No space left on device");
	}

	PVRow get_rows_count() override { return 0; }
};

} // namespace

int main()
{
	pvtest::TestEnv env(filename, fileformat);

	PVFailingOutput output;
	PVRush::PVExtractor extractor(env._format, output, env._sc_file, env._list_inputs);
	PVRush::PVControllerJob_p job = extractor.process_from_agg_nlines(0);

	while (job->running()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	job->cancel();

	PV_VALID(job->running(), false);

	bool failure_reported = false;
	try {
		job->wait_end();
	} catch (output_failure const&) {
		failure_reported = true;
	}
	PV_VALID(failure_reported, true);

	return 0;
}
