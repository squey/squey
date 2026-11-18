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

#ifndef LIBPVPCAP_EXTRACTION_H
#define LIBPVPCAP_EXTRACTION_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pvpcap
{

struct extraction_error : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

//! What extraction::line::stream holds for a frame of a capture that could not be read.
inline constexpr uint32_t NO_STREAM = UINT32_MAX;

/**
 * Has tshark write fields of the frames of a capture, and hands its lines over in the order
 * of the frames.
 *
 * A tshark process reads one packet after the other, which leaves most processors idle. The
 * capture is split instead: the frames of a conversation go to the same tshark, so that it
 * reassembles the conversation as a single one would, and several tshark run at once. As many
 * run as there are processors, fed through pipes while the capture is read, unless they would
 * not fit in memory: a tshark keeps about four times what it read until it ends. A capture
 * too large for that is split in more parts, each written to a file, then read by a tshark
 * that ends with it.
 *
 * A small capture, or one that can't be split (a capture of system calls, a format tshark
 * reads but this does not), is read by a single tshark.
 *
 * Some fields are only written right by a tshark that read the frames after theirs, as
 * Wireshark does once it read the whole capture: such a tshark reads its part twice, which a
 * pipe does not allow. The parts are then written to files first.
 *
 * Whatever the split, the lines come in the order of the frames, and each of them says which
 * frame it is about: the frames a display filter dropped are the only ones missing.
 *
 * Thread-safety: next() is for one thread, cancel() may be called from any.
 */
class extraction
{
  public:
	/**
	 * A field asked for each frame.
	 *
	 * Some fields tshark writes depend on which frames it read: those this rewrites, as a
	 * tshark that read the whole capture would have written them.
	 */
	struct field {
		enum class kind {
			text,                //!< as tshark writes it
			frames,              //!< numbers of frames, such as http.request_in
			absolute_time,       //!< rewritten in UTC, as 2011-05-03T23:43:01.545400000+0000
			time_since_first,    //!< frame.time_relative
			time_since_previous, //!< frame.time_delta
			time_since_shown,    //!< frame.time_delta_displayed
			summary,             //!< the Info column, which names the frames TCP reassembles in
		};
		std::string name;
		kind type = kind::text;
	};

	struct options {
		//! Path of tshark, tshark_path() if empty.
		std::string program;
		//! Of tshark, after the program and before the fields: filters, preferences...
		std::vector<std::string> tshark_options;
		//! Asked for each frame, in that order.
		std::vector<field> fields;
		//! What tshark writes between the occurrences of a field it writes all of.
		std::string aggregator = "|";
		//! Of the tshark processes, on top of the variables of this process.
		std::vector<std::pair<std::string, std::string>> environment;
		//! Where the temporary files go, in a directory of their own, removed with them.
		std::string work_directory;
		//! Tshark processes running at once, 0 for as many as hardware threads.
		size_t workers = 0;
		//! Bytes the tshark processes may take together, 0 for half the available memory.
		uint64_t memory = 0;
		//! A capture below this size is not split.
		uint64_t smallest_split = 8 * 1024 * 1024;
		//! Bytes of the capture below which a part would not be worth a tshark of its own.
		uint64_t smallest_part = 4 * 1024 * 1024;
		/**
		 * Whether tshark reads its part twice (-2), for fields that need the frames after
		 * theirs, such as http.response_in: it takes about a quarter longer.
		 */
		bool two_pass = false;
	};

	struct line {
		uint64_t frame = 0;  //!< number of the frame in the capture, from 1
		uint32_t stream = 0; //!< conversation of the frame, from 0 by order of appearance
		uint64_t bytes = 0;  //!< of the file, read up to this frame since the previous line
		//! What tshark wrote for the fields, separated by SEPARATOR, valid until next().
		std::string_view fields;
	};

	/**
	 * Starts the extraction: tshark processes run from now on.
	 *
	 * @throw extraction_error if the capture can't be read at all
	 */
	extraction(const std::string& capture, const options& opts);

	//! Stops what still runs, and removes the temporary files.
	~extraction();

	extraction(const extraction&) = delete;
	extraction& operator=(const extraction&) = delete;

	/**
	 * Waits for the line of the next frame tshark wrote fields of.
	 *
	 * @return false past the last one, or once canceled
	 * @throw extraction_error if tshark failed, in its own words
	 */
	bool next(line& l);

	//! Stops the extraction: next() returns false from then on.
	void cancel();

	//! Once next() returned false: number of frames of the capture.
	uint64_t frames() const;

	//! Once next() returned false: number of conversations of the capture.
	uint32_t streams() const;

	/**
	 * Once next() returned false: what cut the reading of the capture short, such as its
	 * end in the middle of a packet. Empty if it was read whole.
	 */
	std::string damage() const;

	//! Number of tshark processes the capture is split between.
	size_t parts() const;

  private:
	struct implementation;
	std::unique_ptr<implementation> _impl;
};

} // namespace pvpcap

#endif // LIBPVPCAP_EXTRACTION_H
