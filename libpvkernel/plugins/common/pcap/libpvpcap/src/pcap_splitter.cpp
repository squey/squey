//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <libpvpcap/pcap_splitter.h>
#include <libpvpcap/sniff_def.h>

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <map>
#include <algorithm>
#include <bitset>
#include <unordered_map>
#include <numeric>

#include <QDir>
#include <QFileInfo>
#include <QString>

#include <boost/filesystem.hpp>

#include <tbb/parallel_sort.h>

#include <pvhwloc.h>
#include <pvlogger.h>

static std::pair<in6_addr, in6_addr> srcip_dstip(const sniff_ip* ip, bool ipv4)
{
	in6_addr ip_src;
	in6_addr ip_dst;

	if (ipv4) {
		uint32_t ipv4_src = ip->ip_src.s_addr;
		ip_src.s6_addr32[0] = 0;
		ip_src.s6_addr32[1] = 0;
		ip_src.s6_addr32[2] = 0x0000FFFF;
		ip_src.s6_addr32[3] = ipv4_src;

		uint32_t ipv4_dst = ip->ip_dst.s_addr;
		ip_dst.s6_addr32[0] = 0;
		ip_dst.s6_addr32[1] = 0;
		ip_dst.s6_addr32[2] = 0x0000FFFF;
		ip_dst.s6_addr32[3] = ipv4_dst;
	} else {
		const ip6_hdr* ipv6_h = reinterpret_cast<const ip6_hdr*>(ip);

		ip_src = ipv6_h->ip6_src;
		ip_dst = ipv6_h->ip6_dst;
	}

	return std::make_pair(ip_src, ip_dst);
}

/**
 * @class LRUList
 * A template class that implements a LRU cache with limited size. Each time the user puts an
 * element it goes to head of the
 * list as the most recently used element (if the element was already in the list it advances to the
 * head of the list).
 * The last element in the list is the one least recently used and will be pulled out of the list if
 * it reaches its max size
 * and a new element comes in. All actions on this LRU list are O(1)
 */
template <typename T>
class LRUList
{
  public:
	typedef typename std::list<T>::iterator ListIterator;
	typedef typename std::map<T, ListIterator>::iterator MapIterator;

	/**
	 * A c'tor for this class
	 * @param[in] maxSize The max size this list can go
	 */
	LRUList(size_t maxSize) { m_MaxSize = maxSize; }

	/**
	 * Puts an element in the list. This element will be inserted (or advanced if it already exists)
	 * to the head of the
	 * list as the most recently used element. If the list already reached its max size and the
	 * element is new this method
	 * will remove the least recently used element and return a pointer to it. Method complexity is
	 * O(1)
	 * @param[in] element The element to insert or to advance to the head of the list (if already
	 * exists)
	 * @return A pointer to the element that was removed from the list in case the list already
	 * reached its max size.
	 * If the list didn't reach its max size NULL will be returned. Notice it's the responsibility
	 * of the user to free
	 * this pointer's memory when done using it
	 */
	T* put(const T& element)
	{
		m_CacheItemsList.push_front(element);
		MapIterator iter = m_CacheItemsMap.find(element);
		if (iter != m_CacheItemsMap.end())
			m_CacheItemsList.erase(iter->second);
		m_CacheItemsMap[element] = m_CacheItemsList.begin();

		if (m_CacheItemsList.size() > m_MaxSize) {
			ListIterator lruIter = m_CacheItemsList.end();
			--lruIter;
			T* deletedValue = new T(*lruIter);
			m_CacheItemsMap.erase(*lruIter);
			m_CacheItemsList.erase(lruIter);

			return deletedValue;
		}

		return nullptr;
	}

	/**
	 * Get the most recently used element (the one at the beginning of the list)
	 * @return The most recently used element
	 */
	const T& getMRUElement() { return m_CacheItemsList.front(); }

	/**
	 * Get the least recently used element (the one at the end of the list)
	 * @return The least recently used element
	 */
	const T& getLRUElement() { return m_CacheItemsList.back(); }

  private:
	std::list<T> m_CacheItemsList;
	std::map<T, ListIterator> m_CacheItemsMap;
	size_t m_MaxSize;
};

template <typename T>
union bitset_cast_t {
	using value_type = std::bitset<sizeof(T) * 8>;

	bitset_cast_t(T v) : as_value(v){};
	value_type cast() { return as_bits; }

  private:
	T as_value;
	value_type as_bits;
};

class PacketSplitter
{
  protected:
	// in order to support all OS's, the maximum number of concurrent open file is set to 500
	static const size_t CONCURRENT_OPENED_FILES_COUNT = 500;

  public:
	PacketSplitter(const std::string& input_pcap_fname,
	               const std::string& output_pcap_dir,
	               size_t max_file_count = pvhwloc::thread_count())
	    : _input_pcap_fname(input_pcap_fname)
	    , _output_pcap_dir(output_pcap_dir)
	    , _max_file_count(max_file_count)
	    , _packets_counts(max_file_count, 0)
	{
	}

	virtual ~PacketSplitter()
	{
		// close opened pcap dumpers
		for (auto it : _output_files) {
			pcap_dump_close(it.second);
		}
	}

  public:
	virtual void write_packet(const pcap_pkthdr* header, const u_char* packet)
	{
		auto open_file = [](const std::string& fname, const char* opts) {
			FILE* pcap_file = fopen(fname.c_str(), opts);
			pcap_t* dumpfilehandle = pcap_open_dead(1, 65535);
			return pcap_dump_fopen(dumpfilehandle, pcap_file);
		};

		// get the file number to write the current packet to
		size_t file_idx = get_file_idx();

		// if file number is seen for the first time (meaning it's the first packet written to it)
		if (_output_files.find(file_idx) == _output_files.end()) {
			// open output file
			const std::string& filename =
			    get_filename(_output_pcap_dir, _input_pcap_fname, file_idx);
			_output_files[file_idx] = open_file(filename, "wb");
			_files.emplace_back(filename, _input_pcap_fname);
		}

		// if file number exists in the map but file handler is null it means this file was
		// open once and then closed. In this case we need to re-open the output file in
		// append mode
		else if (_output_files[file_idx] == nullptr) {
			// re-open output file
			const std::string& filename =
			    get_filename(_output_pcap_dir, _input_pcap_fname, file_idx);
			_output_files[file_idx] = open_file(filename, "ab");
		}

		// write packet
		pcap_dump((u_char*)_output_files[file_idx], header, packet);
		_packets_counts[file_idx]++;

		if (_max_file_count > CONCURRENT_OPENED_FILES_COUNT) {
			for (size_t file_idx : _files_idx_to_close) {
				if (_output_files.find(file_idx) != _output_files.end()) {
					pcap_dump_close(_output_files[file_idx]);
					_output_files[file_idx] = nullptr;
				}
			}
		}
	}

	virtual void finished()
	{
		for (size_t file_idx = 0; file_idx < _files.size(); file_idx++) {
			_files[file_idx].set_packets_count(_packets_counts[file_idx]);
		}
	}

	pvpcap::splitted_files_t files() { return _files; }

  protected:
	static std::string get_filename(const std::string& output_pcap_dir,
	                         const std::string& input_pcap_filename,
	                         size_t file_idx)
	{
		std::string output_pcap_filename =
		    output_pcap_dir + std::string(1, '/') +
		    boost::filesystem::path(input_pcap_filename).stem().string() + "-";

		std::ostringstream sstream;
		sstream << std::setw(4) << std::setfill('0') << file_idx;
		std::string filename = output_pcap_filename.c_str() + sstream.str() + ".pcap";

		return filename;
	}

  private:
	size_t get_file_idx() { return _packets_count++ % _max_file_count; }

  protected:
	std::string _input_pcap_fname;
	std::string _output_pcap_dir;
	size_t _max_file_count = 0;
	size_t _next_file_idx = 0;
	std::vector<uint32_t> _packets_counts;
	uint32_t _packets_count = 0;
	// a map of file number to pcap dumper
	std::unordered_map<size_t, pcap_dumper_t*> _output_files;
	pvpcap::splitted_files_t _files;
	std::vector<size_t> _files_idx_to_close;
};

/**
 * A splitter which dispatch and write packets to the specified number of pcap files
 * according to the packet flow (5-tuple)
 */
class FlowSplitter : public PacketSplitter
{

  private:
	struct stream_infos_t {
		stream_infos_t(size_t r, size_t h, time_t t) : rel_packet_index(r), hash(h), timestamp(t) {}
		size_t rel_packet_index;
		size_t hash;
		time_t timestamp;
	};

  private:
	using flow_t = std::tuple<uint16_t, // protocol
	                          in6_addr, // ip_source
	                          in6_addr, // ip_dest
	                          uint16_t, // ip_source_port
	                          uint16_t  // ip_dest_port
	                          >;

  public:
	FlowSplitter(const std::string& input_pcap_fname,
	             const std::string& output_pcap_dir,
	             size_t max_file_count = pvhwloc::thread_count() * 10)
	    : PacketSplitter(input_pcap_fname, output_pcap_dir, max_file_count)
	    , _LRU_file_list(CONCURRENT_OPENED_FILES_COUNT)
	{
	}

  public:
	void finished() override
	{
		// compute streams ids
		_streams_ids.resize(_flow_table.size());
		std::vector<uint32_t> frame_numbers(_stream_infos.size());
		std::iota(frame_numbers.begin(), frame_numbers.end(), 0);
		tbb::parallel_sort(
		    frame_numbers.begin(), frame_numbers.end(), [&](uint32_t idx1, uint32_t idx2) {
			    if (_stream_infos[idx1].hash != _stream_infos[idx2].hash) {
				    return _stream_infos[idx1].hash < _stream_infos[idx2].hash;
			    } else {
				    return _stream_infos[idx1].timestamp < _stream_infos[idx2].timestamp;
			    }
			});

		size_t stream_id = 0;
		size_t file_idx = _flow_table[_stream_infos[frame_numbers[0]].hash];
		_streams_ids[file_idx].reset(
		    new pvpcap::splitted_file_t::vector_uint32_t(_packet_indexes[file_idx]->size()));
		(*_streams_ids[file_idx])[_stream_infos[frame_numbers[0]].rel_packet_index] = stream_id;
		for (size_t i = 1; i < _stream_infos.size(); i++) {
			const size_t hash = _stream_infos[frame_numbers[i]].hash;
			const size_t prev_hash = _stream_infos[frame_numbers[i - 1]].hash;
			// static constexpr const time_t TIMEOUT = 72;
			// const size_t timestamp = _stream_infos[frame_numbers[i]].timestamp;
			// const size_t prev_timestamp = _stream_infos[frame_numbers[i-1]].timestamp;

			if (hash != prev_hash /*or ((timestamp - prev_timestamp) > TIMEOUT)*/) {
				stream_id++;

				//if (hash != prev_hash) {
					file_idx = _flow_table[hash];
					if (_streams_ids[file_idx].get() == nullptr) {
						_streams_ids[file_idx].reset(new pvpcap::splitted_file_t::vector_uint32_t(
						    _packet_indexes[file_idx]->size()));
					}
				//}
			}

			(*_streams_ids[file_idx])[_stream_infos[frame_numbers[i]].rel_packet_index] = stream_id;
		}

		// save packets indexes order and streams ids
		for (size_t file_idx = 0; file_idx < _packet_indexes.size(); file_idx++) {
			_files[file_idx].set_packets_indexes(_packet_indexes[file_idx]);
			_files[file_idx].set_streams_ids(_streams_ids[file_idx], stream_id + 1);
		}
	}

  public:
	void write_packet(const pcap_pkthdr* header, const u_char* packet) override
	{
		auto open_file = [](const std::string& fname, const char* opts) {
			FILE* pcap_file = fopen(fname.c_str(), opts);
			pcap_t* dumpfilehandle = pcap_open_dead(1, 65535);
			return pcap_dump_fopen(dumpfilehandle, pcap_file);
		};

		stream_infos_t f = extract_flow(header, packet);

		// get the file number to write the current packet to
		size_t file_idx = get_file_idx(f.hash);

		// if file number is seen for the first time (meaning it's the first packet written to it)
		if (_output_files.find(file_idx) == _output_files.end()) {
			// open output file
			const std::string& filename =
			    get_filename(_output_pcap_dir, _input_pcap_fname, file_idx);
			_output_files[file_idx] = open_file(filename, "wb");
			_files.emplace_back(filename, _input_pcap_fname);
			_packet_indexes.emplace_back(new pvpcap::splitted_file_t::vector_uint32_t);
		}

		// if file number exists in the map but file handler is null it means this file was
		// open once and then closed. In this case we need to re-open the output file in
		// append mode
		else if (_output_files[file_idx] == nullptr) {
			// re-open output file
			const std::string& filename =
			    get_filename(_output_pcap_dir, _input_pcap_fname, file_idx);
			_output_files[file_idx] = open_file(filename, "ab");
		}

		// write packet
		pcap_dump((u_char*)_output_files[file_idx], header, packet);

		// save packet index and sequence number
		_packet_indexes[file_idx]->emplace_back(_packets_count++);
		f.rel_packet_index = _packet_indexes[file_idx]->size() - 1;
		_stream_infos.emplace_back(f);

		if (_max_file_count > CONCURRENT_OPENED_FILES_COUNT) {
			for (size_t file_idx : _files_idx_to_close) {
				if (_output_files.find(file_idx) != _output_files.end()) {
					pcap_dump_close(_output_files[file_idx]);
					_output_files[file_idx] = nullptr;
				}
			}
		}
	}

  private:
	size_t get_file_idx(size_t hash)
	{
		// if flow isn't found in the flow table
		if (_flow_table.find(hash) == _flow_table.end()) {
			// create a new entry and get a new file number for it
			_flow_table[hash] = get_next_file_idx();
		} else { // flow is found in the flow table
			// indicate file is being written because this file may not be in the LRU list (and
			// hence closed),
			// so we need to put it there, open it, and maybe close another file
			size_t file_idx = _flow_table[hash];
			if (_max_file_count > CONCURRENT_OPENED_FILES_COUNT) {
				size_t* splitted_file_to_close = _LRU_file_list.put(file_idx);
				if (splitted_file_to_close != nullptr) {
					_files_idx_to_close.push_back(*splitted_file_to_close);
					delete splitted_file_to_close;
				}
			}
		}

		return _flow_table[hash];
	}

	size_t get_next_file_idx()
	{
		size_t next_file_idx = 0;

		// zero _max_file_count means no limit
		if (_max_file_count <= 0) {
			next_file_idx = _next_file_idx++;
		} else { // _max_file_count is positive, meaning there is a output file limit
			next_file_idx = (_next_file_idx++) % _max_file_count;
		}

		// put the next file in the LRU list
		if (_max_file_count > CONCURRENT_OPENED_FILES_COUNT) {
			size_t* file_idx_to_close = _LRU_file_list.put(next_file_idx);
			if (file_idx_to_close != nullptr) {
				// if a file is pulled out of the LRU list - return it
				_files_idx_to_close.push_back(*file_idx_to_close);
				delete file_idx_to_close;
			}
		}
		return next_file_idx;
	}

	/**
	 * Return a hash from a 5-tuple
	 */
	stream_infos_t extract_flow(const pcap_pkthdr* header, const u_char* packet)
	{
		uint16_t protocol = (uint8_t)IPPROTO_MAX;
		in6_addr ip_src = {{ .__u6_addr32 = { 0, 0, 0, 0 }}};
		in6_addr ip_dst = {{ .__u6_addr32 = { 0, 0, 0, 0 }}};
		uint16_t ip_src_port = 0;
		uint16_t ip_dst_port = 0;
		time_t timestamp = header->ts.tv_sec;

		size_t size_ip = 0;

		const ethhdr* eth_h = reinterpret_cast<const ethhdr*>(packet);
		uint16_t eth_proto = ntohs(eth_h->h_proto);

		if (eth_proto == ETH_P_IPV6) {
			const sniff_ip* ip = reinterpret_cast<const sniff_ip*>(packet + sizeof(ethhdr));
			size_ip = IP_HL(ip) * 4;
			protocol = ip->ip_p;
			std::tie(ip_src, ip_dst) = srcip_dstip(ip, false);
		} else if (eth_proto == ETH_P_IP) {
			const sniff_ip* ip = reinterpret_cast<const sniff_ip*>(packet + sizeof(ethhdr));
			size_ip = IP_HL(ip) * 4;
			protocol = ip->ip_p;
			std::tie(ip_src, ip_dst) = srcip_dstip(ip, true);
		} else if (eth_proto == ETH_P_8021Q) { // 802.1Q VLAN Extended Header
			const sniff_ip* ipq = reinterpret_cast<const sniff_ip*>(packet + sizeof(ethhdr) + 4);
			size_ip = IP_HL(ipq) * 4;
			protocol = ipq->ip_p;
			std::tie(ip_src, ip_dst) = srcip_dstip(ipq, IP_V(ipq) == 4);
		} else {
			protocol = eth_proto;
		}

		if (protocol == IPPROTO_TCP) {
			const sniff_tcp* tcp =
			    reinterpret_cast<const sniff_tcp*>(packet + sizeof(ethhdr) + size_ip);

			ip_src_port = ntohs(tcp->th_sport);
			ip_dst_port = ntohs(tcp->th_dport);
		} else if (protocol == IPPROTO_UDP) {
			const sniff_udp* udp =
			    reinterpret_cast<const sniff_udp*>(packet + sizeof(ethhdr) + size_ip);

			ip_src_port = ntohs(udp->sport);
			ip_dst_port = ntohs(udp->dport);
		}

		union half_ipv6 {
			half_ipv6(uint32_t h1, uint32_t h2)
			{
				u32[0] = h1;
				u32[1] = h2;
			}
			uint32_t u32[2];
			uint64_t u64;
		};

		// merge symetric directions flows
		uint64_t ip_src_lo = half_ipv6(ip_src.s6_addr32[0], ip_src.s6_addr32[1]).u64;
		uint64_t ip_src_hi = half_ipv6(ip_src.s6_addr32[2], ip_src.s6_addr32[3]).u64;
		uint64_t ip_dst_lo = half_ipv6(ip_dst.s6_addr32[0], ip_dst.s6_addr32[1]).u64;
		uint64_t ip_dst_hi = half_ipv6(ip_dst.s6_addr32[2], ip_dst.s6_addr32[3]).u64;
		if (ip_src_lo < ip_dst_lo or (ip_src_lo == ip_dst_lo and ip_src_hi < ip_dst_hi)) {
			std::swap(ip_src, ip_dst);
			std::swap(ip_src_port, ip_dst_port);
		}

		// hash the 5-tuple and look for it in the flow table
		using bitset_t = bitset_cast_t<flow_t>;
		flow_t flow = std::make_tuple(protocol, ip_src, ip_dst, ip_src_port, ip_dst_port);
		size_t hash = std::hash<bitset_t::value_type>()(bitset_t(flow).cast());

		return {0, hash, timestamp};
	}

  private:
	LRUList<size_t> _LRU_file_list;
	// A flow table that keeps track of all flows (a flow is usually identified by 5-tuple)
	std::unordered_map<size_t, size_t> _flow_table;
	std::vector<pvpcap::splitted_file_t::vector_uint32_sp> _packet_indexes;
	std::vector<stream_infos_t> _stream_infos;
	std::vector<pvpcap::splitted_file_t::vector_uint32_sp> _streams_ids;
};

namespace pvpcap
{

splitted_files_t
split_pcaps(const std::vector<std::string>& input_pcap_filenames,
            const std::string& output_pcap_dir,
            bool preserve_flows,
            bool& canceled,
            const std::function<void(size_t total_datasize)>& f_total_datasize /* = {} */,
            const std::function<void(size_t current_datasize)>& f_progression /* = {} */)
{
	splitted_files_t files;

	size_t total_datasize = 0;
	for (const std::string& input_pcap_filename : input_pcap_filenames) {
		total_datasize +=
		    std::ifstream(input_pcap_filename, std::ifstream::ate | std::ifstream::binary).tellg();
	}
	if (f_total_datasize) {
		f_total_datasize(total_datasize);
	}

	size_t current_datasize = 0;
	for (const std::string& input_pcap_filename : input_pcap_filenames) {
		splitted_files_t f = split_pcap(input_pcap_filename, output_pcap_dir, preserve_flows,
		                                canceled, [&](size_t s) {
			                                current_datasize += s;
			                                f_progression(current_datasize);
			                            });
		std::move(f.begin(), f.end(), back_inserter(files));
	}
	if (f_progression) {
		f_progression(total_datasize);
	}

	// sort by decreasing packets count
	std::sort(files.begin(), files.end(), [](const auto& f1, const auto& f2) {
		return f1.packets_count() > f2.packets_count();
	});

	return files;
}

splitted_files_t
split_pcap(const std::string& input_pcap_filename,
           const std::string& output_dir,
           bool preserve_flows,
           bool& canceled,
           const std::function<void(size_t current_datasize)>& f_progression /* = {} */)
{
	static constexpr const size_t PACKET_PROGRESS_THRESHOLD_SIZE = 1 * 1024 * 1024;

	QString output_pcap_dir =
	    QString::fromStdString(output_dir) + "/" +
	    QFileInfo(QString::fromStdString(input_pcap_filename) + ".dir").fileName();
	QDir().mkpath(output_pcap_dir);

	std::unique_ptr<PacketSplitter> splitter;
	if (preserve_flows) {
		splitter.reset(new FlowSplitter(input_pcap_filename, output_pcap_dir.toStdString()));
	} else {
		splitter.reset(new PacketSplitter(input_pcap_filename, output_pcap_dir.toStdString()));
	}

	// open input pcap file for reading
	char error_buffer[PCAP_ERRBUF_SIZE];
	pcap_t* input_pcap_handle = pcap_open_offline(input_pcap_filename.c_str(), error_buffer);
	if (not input_pcap_handle) {
		pvlogger::error() << "Error opening input pcap file :" << input_pcap_filename << std::endl;
		return {};
	}

	size_t aggregated_packet_size = 0;

	// read all packets from input file
	pcap_pkthdr* header;
	const u_char* packet;
	while (int ret = pcap_next_ex(input_pcap_handle, &header, &packet) >= 0 and not canceled) {
		if (ret < 0) {
			pcap_perror(input_pcap_handle, error_buffer);
			continue;
		}
		// write packet to proper output pcap file according to packet flow (5-tuple)
		splitter->write_packet(header, packet);

		// notify splitting progress
		aggregated_packet_size += header->caplen;
		if (aggregated_packet_size > PACKET_PROGRESS_THRESHOLD_SIZE) {
			if (f_progression) {
				f_progression(aggregated_packet_size);
				aggregated_packet_size = 0;
			}
		}
	}

	splitter->finished();
	pcap_close(input_pcap_handle);

	splitted_files_t files = splitter->files();

	// sort by decreasing file size
	std::sort(files.begin(), files.end(), [](const auto& f1, const auto& f2) {
		return f1.packets_count() > f2.packets_count();
	});

	return files;
}

} // namespace pvpcap
