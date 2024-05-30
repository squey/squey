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

#ifndef __PVPCAPSOURCE_FILE_H__
#define __PVPCAPSOURCE_FILE_H__

#include <iterator>
#include <fcntl.h>
#include <memory>

#include <QString>

#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVUnicodeSource.h>

#include "PVPcapSource.h"
#include "../../common/pcap/PVPcapDescription.h"
#include "../../common/pcap/libpvpcap/include/libpvpcap.h"

namespace PVRush
{

class PVPcapSource : public PVUnicodeSource<>
{
  public:
	PVPcapSource(PVInput_p input,
	             PVPcapDescription* input_desc,
	             size_t chunk_size,
	             const alloc_chunk& alloc = alloc_chunk())
	    : PVUnicodeSource<>(input, chunk_size, alloc), _input_desc(input_desc)
	{
		_original_pcap_filename =
		    QFileInfo(input_desc->original_pcap_path()).fileName().toStdString();
	}

  public:
	void add_element(char* begin, char* end) override
	{
		size_t element_size = std::distance(begin, end);

		const std::string& global_frame_number =
		    std::to_string(_input_desc->packets_indexes()[_packet_count] +
		                   _input_desc->pcap_packets_count_offset() + 1) +
		    pvpcap::SEPARATOR;
		const std::string& stream_id = std::to_string(_input_desc->streams_ids()[_packet_count] +
		                                              _input_desc->pcap_streams_id_offset()) +
		                               pvpcap::SEPARATOR;

		size_t new_size = element_size + global_frame_number.size() + stream_id.size() +
		                  (_input_desc->multi_inputs() ? (_original_pcap_filename.size() + 1) : 0);
		PVCore::PVElement* new_element = PVUnicodeSource<>::add_uninitialized_element(new_size);

		std::copy(global_frame_number.begin(), global_frame_number.end(), new_element->begin());

		std::copy(stream_id.begin(), stream_id.end(),
		          new_element->begin() + global_frame_number.size());

		size_t len = 0;
		if (_input_desc->multi_inputs()) {
			std::copy(_original_pcap_filename.begin(), _original_pcap_filename.end(),
			          new_element->begin() + global_frame_number.size() + stream_id.size());
			*(new_element->begin() + (global_frame_number.size() + stream_id.size() +
			                          _original_pcap_filename.size())) = pvpcap::SEPARATOR[0];
			len = _original_pcap_filename.size() + 1;
		}
		std::copy(begin, end,
		          new_element->begin() + global_frame_number.size() + stream_id.size() + len);

		_packet_count++;
	}

	/**
	 * Delete temporary CSV and packets_order files
	 */
	void release_input(bool cancel_first = false) override
	{
		std::remove(_input_desc->path().toStdString().c_str());

		_input_desc->clear_packets_indexes();
		_input_desc->reset_streams_ids();

		PVUnicodeSource<>::release_input(cancel_first);
	}

  private:
	PVPcapDescription* _input_desc = nullptr;
	size_t _packet_count = 0;

	std::string _original_pcap_filename;
};
} // namespace PVRush

#endif // __PVPCAPSOURCE_FILE_H__
