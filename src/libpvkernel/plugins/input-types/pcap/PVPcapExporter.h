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

#ifndef __PVCORE_PVPCAPEXPORTER_H__
#define __PVCORE_PVPCAPEXPORTER_H__

#include <QDesktopServices>
#include <QFile>
#include <QUrl>

#include "../../common/pcap/PVPcapDescription.h"
#include "../../common/pcap/libpvpcap/include/libpvpcap/capture.h"

#include <pvcop/db/algo.h>
#include <pvkernel/rush/PVExporter.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVNraw.h>

#include <algorithm>
#include <map>

namespace PVRush
{

/**
 * Writes the packets of the selected rows, as their captures hold them.
 *
 * The packets are found by their number, which the first column holds: whatever the order of
 * the rows, and whether or not a display filter left frames out of the import.
 */
class PVPcapExporter : public PVExporterBase
{
	static constexpr size_t PACKETS_STEP_COUNT = 10 * 1024;

	//! Columns the import puts first.
	static constexpr PVCol FRAME_COLUMN{0};
	static constexpr PVCol STREAM_COLUMN{1};

	struct capture {
		std::string path;
		uint64_t frames_offset;
	};

  public:
	PVPcapExporter(const PVRush::PVInputType::list_inputs& inputs, PVRush::PVNraw const& nraw)
	    : _inputs(inputs), _nraw(nraw)
	{
	}

	/**
	 * Whether a capture holding these inputs has to be pcapng: when one of them is, or when
	 * their pcap headers differ, which no single pcap header can tell of.
	 */
	static bool needs_pcapng(const PVRush::PVInputType::list_inputs& inputs)
	{
		std::vector<uint8_t> header;
		for (const capture& c : captures(inputs)) {
			try {
				pvpcap::capture_reader reader(c.path);
				if (reader.file_format() == pvpcap::capture_reader::format::pcapng) {
					return true;
				}
				// The snapshot length aside, which the largest of them gives: bytes 16 to 19 of
				// the 24 of a pcap header.
				std::vector<uint8_t> h(reader.header().begin(), reader.header().end());
				if (h.size() >= 20) {
					std::fill(h.begin() + 16, h.begin() + 20, 0);
				}
				if (not header.empty() and h != header) {
					return true;
				}
				header = std::move(h);
			} catch (const pvpcap::capture_error&) {
				// Said when exporting, once it is asked for.
			}
		}
		return false;
	}

  public:
	void export_rows(const std::string& file_path, const PVCore::PVSelBitField& sel) override
	{
		PVCore::PVSelBitField selection(sel);
		if (_export_complete_stream) {
			selection = complete_streams(sel);
		}
		const std::vector<uint64_t> wanted = wanted_frames(selection);

		try {
			export_frames(file_path, wanted, selection.bit_count());
		} catch (const pvpcap::capture_error& e) {
			QFile::remove(QString::fromStdString(file_path));
			throw PVExportError(e.what());
		}
		if (_canceled) {
			_canceled = false;
			QFile::remove(QString::fromStdString(file_path));
			return;
		}
		if (_open_pcap_after_export) {
			QDesktopServices::openUrl(QUrl::fromLocalFile(QString::fromStdString(file_path)));
		}
	}

  public:
	bool get_export_complete_stream() const { return _export_complete_stream; }
	bool get_open_pcap_after_export() const { return _open_pcap_after_export; }
	void set_export_complete_stream(bool export_complete_stream)
	{
		_export_complete_stream = export_complete_stream;
	}
	void set_open_pcap_after_export(bool open_pcap_after_export)
	{
		_open_pcap_after_export = open_pcap_after_export;
	}

  private:
	//! The captures, each once, by the number of their first frame.
	static std::vector<capture> captures(const PVRush::PVInputType::list_inputs& inputs)
	{
		// An investigation saved by an older version has an input per part of each capture.
		std::map<std::string, uint64_t> offsets;
		for (const auto& input : inputs) {
			const auto* description = dynamic_cast<const PVPcapDescription*>(input.get());
			if (description == nullptr) {
				continue;
			}
			const auto known = description->offsets();
			offsets.emplace(description->path().toStdString(), known ? known->first : 0);
		}
		std::vector<capture> result;
		for (const auto& [path, offset] : offsets) {
			result.push_back({path, offset});
		}
		std::sort(result.begin(), result.end(), [](const capture& a, const capture& b) {
			return a.frames_offset < b.frames_offset;
		});
		return result;
	}

	//! The rows of the conversations the selected rows belong to.
	PVCore::PVSelBitField complete_streams(const PVCore::PVSelBitField& sel) const
	{
		const pvcop::db::array& streams = _nraw.column(STREAM_COLUMN);
		pvcop::db::array distinct_values;
		PVCore::PVSelBitField outsel(sel.count());
		outsel.select_none();
		pvcop::db::algo::distinct(streams, distinct_values, sel);
		pvcop::db::algo::subselect(streams, distinct_values, {}, outsel);
		return outsel;
	}

	//! A bit for each frame number, set for the frames of the selected rows.
	std::vector<uint64_t> wanted_frames(const PVCore::PVSelBitField& sel) const
	{
		const pvcop::db::array& column = _nraw.column(FRAME_COLUMN);
		const auto& numbers = column.to_core_array<uint32_t>();
		std::vector<uint64_t> wanted;
		for (size_t row = 0; row < column.size(); row++) {
			if (not sel.get_line(row) or not column.is_valid(row)) {
				continue;
			}
			const uint64_t frame = numbers[row];
			if (frame / 64 >= wanted.size()) {
				wanted.resize(frame / 64 + 1);
			}
			wanted[frame / 64] |= uint64_t(1) << (frame % 64);
		}
		return wanted;
	}

	void export_frames(const std::string& file_path,
	                   const std::vector<uint64_t>& wanted,
	                   size_t selected)
	{
		const std::vector<capture> all = captures(_inputs);
		const bool pcapng = needs_pcapng(_inputs);
		const auto is_wanted = [&](uint64_t frame) {
			return frame / 64 < wanted.size() and (wanted[frame / 64] >> (frame % 64)) & 1;
		};

		pvpcap::capture_writer writer(file_path);
		bool header_written = false;
		size_t exported = 0;
		for (const capture& c : all) {
			const uint64_t first = c.frames_offset + 1;
			if (first / 64 >= wanted.size()) {
				continue;
			}
			pvpcap::capture_reader reader(c.path);
			if (pcapng and reader.file_format() == pvpcap::capture_reader::format::pcap) {
				// A section of its own, whose only interface is the link of the capture.
				writer.write(pvpcap::pcapng::section_header());
				writer.write(pvpcap::pcapng::interface_description(
				    reader.link_type(), snap_length(reader), reader.nanoseconds()));
			} else if (not pcapng and not header_written) {
				std::vector<uint8_t> header(reader.header().begin(), reader.header().end());
				set_snap_length(header, reader.swapped(), all);
				writer.write(header);
				header_written = true;
			}

			uint64_t frame = c.frames_offset;
			pvpcap::capture_record record;
			while (reader.next(record)) {
				if (record.type == pvpcap::capture_record::kind::metadata) {
					// Interfaces and secrets the packets of the section need.
					writer.write(record.bytes);
					continue;
				}
				if (not is_wanted(++frame)) {
					continue;
				}
				if (pcapng and reader.file_format() == pvpcap::capture_reader::format::pcap) {
					writer.write(pvpcap::pcapng::packet_of_pcap(record.bytes, reader.swapped(),
					                                            reader.nanoseconds()));
				} else {
					writer.write(record.bytes);
				}
				if (++exported % PACKETS_STEP_COUNT == 0) {
					if (_f_progress) {
						_f_progress(std::min(exported, selected));
					}
					if (_canceled) {
						return;
					}
				}
			}
		}
		writer.close();
		if (_f_progress) {
			_f_progress(std::min(exported, selected));
		}
	}

	static uint32_t snap_length(const pvpcap::capture_reader& reader)
	{
		uint32_t v;
		std::memcpy(&v, reader.header().data() + 16, sizeof(v));
		return reader.swapped() ? std::byteswap(v) : v;
	}

	//! The largest snapshot length of the captures, which the header has to allow for.
	static void
	set_snap_length(std::vector<uint8_t>& header, bool swapped, const std::vector<capture>& all)
	{
		uint32_t largest = 0;
		for (const capture& c : all) {
			try {
				largest = std::max(largest, snap_length(pvpcap::capture_reader(c.path)));
			} catch (const pvpcap::capture_error&) {
			}
		}
		const uint32_t stored = swapped ? std::byteswap(largest) : largest;
		std::memcpy(header.data() + 16, &stored, sizeof(stored));
	}

  protected:
	const PVRush::PVInputType::list_inputs& _inputs;
	PVRush::PVNraw const& _nraw;
	bool _export_complete_stream = true;
	bool _open_pcap_after_export = true;
};

} // namespace PVRush

#endif // __PVCORE_PVEXPORTER_H__
