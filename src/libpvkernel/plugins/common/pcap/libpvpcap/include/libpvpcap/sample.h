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

#ifndef LIBPVPCAP_SAMPLE_H
#define LIBPVPCAP_SAMPLE_H

#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pvpcap
{

//! What the first frames of a capture tell of the whole of it, read without tshark.
struct capture_overview {
	std::string format;            //!< "pcap" or "pcapng", empty if only tshark reads it
	bool compressed = false;       //!< such as a .pcap.gz
	uint64_t size = 0;             //!< of the file, in bytes
	uint64_t frames = 0;           //!< read, all those of the capture if `whole`
	bool whole = false;            //!< whether the capture has no other frame
	uint64_t estimated_frames = 0; //!< in the whole capture, out of the room the read ones take
	int64_t first_timestamp = 0;   //!< of the first packet, in nanoseconds since 1970, UTC
	int64_t last_timestamp = 0;    //!< of the last packet read, or 0 if none was
	std::string damage;            //!< what stopped the reading before the end of the file
};

/**
 * Reads the first frames of a capture, as far as it can without tshark: not at all the
 * formats only tshark reads.
 *
 * @param most_frames the more frames, the closer the estimate
 */
capture_overview overview_capture(const std::string& capture, uint64_t most_frames);

//! Which protocols and fields the first frames of a capture hold.
struct capture_sample {
	uint64_t frames = 0; //!< tshark read

	struct protocol {
		std::string name;    //!< filter name, such as "http"
		uint64_t frames = 0; //!< holding it
		uint32_t depth = 0;  //!< least number of protocols below it in a frame
	};
	//! In the order they first appear in.
	std::vector<protocol> protocols;

	struct field {
		uint64_t frames = 0; //!< holding it
		std::string example; //!< its first value, shortened, on a single line
	};
	//! By filter name: some of those are not fields, such as subtrees of tshark.
	std::unordered_map<std::string, field> fields;
};

/**
 * Has tshark write every field of the first frames of a capture, which takes it about a
 * tenth of a second for 1,000 of them.
 *
 * @param tshark_options of how tshark reads the capture, such as -o, -N and -C
 * @param canceled stops tshark once set: what it wrote then is sampled
 * @throw extraction_error if tshark can't read the capture, in its words
 */
capture_sample sample_capture(const std::string& capture,
                              uint64_t frames,
                              const std::vector<std::string>& tshark_options,
                              const std::atomic<bool>& canceled);

} // namespace pvpcap

#endif // LIBPVPCAP_SAMPLE_H
