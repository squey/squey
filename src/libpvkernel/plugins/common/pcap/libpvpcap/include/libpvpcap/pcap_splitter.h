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

#ifndef __LIBPVPCAP_PCPP_H__
#define __LIBPVPCAP_PCPP_H__

#include <stddef.h>
#include <stdint.h>
#include <functional>
#include <vector>
#include <memory>
#include <string>

namespace pvpcap
{

class splitted_file_t
{
  public:
	using vector_uint32_t = std::vector<uint32_t>;
	using vector_uint32_sp = std::shared_ptr<vector_uint32_t>;
	// need a shared_ptr as tbb::pipeline makes object copies

  public:
	splitted_file_t(std::string p, std::string o)
	    : _path(p), _original_pcap_path(o), _streams_ids_count(0)
	{
	}
	splitted_file_t() = default;

  public:
	std::string path() const { return _path; }
	std::string& path() { return _path; }

	void set_packets_count(size_t packets_count) { _packets_count = packets_count; }
	size_t packets_count() const { return _packets_count; }
	std::string original_pcap_path() const { return _original_pcap_path; }
	vector_uint32_sp& packets_indexes() { return _packets_indexes; }
	void set_packets_indexes(pvpcap::splitted_file_t::vector_uint32_sp pi)
	{
		_packets_indexes = pi;
		_packets_count = pi->size();
	}
	size_t streams_ids_count() const { return _streams_ids_count; }
	vector_uint32_sp& streams_ids() { return _streams_ids; }
	void set_streams_ids(pvpcap::splitted_file_t::vector_uint32_sp si, size_t sic)
	{
		_streams_ids = si;
		_streams_ids_count = sic;
	}

  private:
	std::string _path;
	vector_uint32_sp _packets_indexes;
	size_t _packets_count = 0;
	vector_uint32_sp _streams_ids;
	std::string _original_pcap_path;
	size_t _streams_ids_count = 0;
};
using splitted_files_t = std::vector<splitted_file_t>;

splitted_files_t
split_pcaps(const std::vector<std::string>& input_pcap_filenames,
            const std::string& output_pcap_dir,
            bool preserve_flows,
            bool& canceled,
            const std::function<void(size_t total_datasize)>& f_total_datasize = {},
            const std::function<void(size_t current_datasize)>& f_progression = {});
splitted_files_t split_pcap(const std::string& input_pcap_filename,
                            const std::string& output_pcap_dir,
                            bool preserve_flows,
                            bool& canceled,
                            const std::function<void(size_t current_datasize)>& f_progression = {});

} // namespace pvpcap

#endif // __LIBPVPCAP_PCPP_H__
