/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2017
 */
#ifndef __PVCORE_PVPCAPEXPORTER_H__
#define __PVCORE_PVPCAPEXPORTER_H__

#include <QDesktopServices>
#include <QCheckBox>

#include "pcap/PVPcapDescription.h"
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/core/PVSpinBoxType.h>
#include "pcap/PVInputTypePcap.h"

#include <pcap/pcap.h>
#include <light_pcapng_ext.h>

#include <pvcop/db/algo.h>
#include <pvkernel/rush/PVExporter.h>

static PVCore::PVSelBitField complete_streams(const pvcop::db::array& col_in,
                                              const PVCore::PVSelBitField& sel)
{
	pvcop::db::array distinct_values;
	PVCore::PVSelBitField outsel(sel.count());
	outsel.select_none();

	pvcop::db::algo::distinct(col_in, distinct_values, sel);
	pvcop::db::algo::subselect(col_in, distinct_values, {}, outsel);

	return outsel;
}

namespace PVRush
{

class PVPcapExporter : public PVExporterBase
{
  protected:
	static constexpr const size_t PACKETS_STEP_COUNT = 10 * 1024;

  public:
	PVPcapExporter(const PVRush::PVInputType::list_inputs& inputs, PVRush::PVNraw const& nraw)
	    : _inputs(inputs), _nraw(nraw)
	{
	}

  public:
	void export_rows(const std::string& file_path, const PVCore::PVSelBitField& sel)
	{
		std::map<std::string, size_t> pcap_packets_count_offsets;
		for (const auto& input_type_desc : _inputs) {
			PVRush::PVPcapDescription* fd =
			    dynamic_cast<PVRush::PVPcapDescription*>(input_type_desc.get());
			pcap_packets_count_offsets[fd->original_pcap_path().toStdString()] =
			    fd->pcap_packets_count_offset();
		}
		std::vector<std::pair<std::string, size_t>> pcap_packets_count;
		for (const auto& pcap : pcap_packets_count_offsets) {
			_input_pcaps.emplace_back(pcap.first, pcap.second);
		}
		for (size_t i = 0; i < _input_pcaps.size() - 1; i++) {
			_input_pcaps[i].second = _input_pcaps[i + 1].second - _input_pcaps[i].second;
		}
		if (_input_pcaps.size() > 1) {
			_input_pcaps[_input_pcaps.size() - 1].second =
			    sel.count() - _input_pcaps[_input_pcaps.size() - 2].second;
		} else {
			_input_pcaps[0].second = sel.count();
		}

		const pvcop::db::array& column = _nraw.column(PVCol(0));
		_sorted_indexes = column.parallel_sort();

		export_pcap(file_path, sel);
		if (_open_pcap_after_export) {
			QDesktopServices::openUrl(QUrl(file_path.c_str()));
		}
	}

  private:
	virtual void export_pcap(const std::string& file_path, const PVCore::PVSelBitField& sel)
	{
		pvcop::core::array<pvcop::db::index_t> sort_order = _sorted_indexes.to_core_array();

		// output pcap
		pcap_t* dumpfilehandle = pcap_open_dead(1, 65535);
		FILE* pcap_file = fopen(file_path.c_str(), "wb");
		pcap_dumper_t* dumpfile = pcap_dump_fopen(dumpfilehandle, pcap_file);

		char error_buffer[PCAP_ERRBUF_SIZE];
		pcap_pkthdr* header;
		const u_char* packet;

		size_t packet_index = 0;
		size_t exported_packets_count = 0;
		size_t start = 0;
		size_t end = 0;

		for (const std::pair<std::string, size_t>& p : _input_pcaps) {

			const std::string& pcap_path = p.first;
			size_t pcap_packets_count = p.second;

			end += pcap_packets_count;

			if (sel.bit_count(start, end - 1) != 0) {

				pcap_t* input_handle = pcap_open_offline(pcap_path.c_str(), error_buffer);

				while (pcap_next_ex(input_handle, &header, &packet) >= 0) {
					if (sel.get_line(sort_order[packet_index++])) {
						pcap_dump((u_char*)dumpfile, header, packet);

						if (exported_packets_count++ % PACKETS_STEP_COUNT == 0) {
							if (_f_progress) {
								_f_progress(exported_packets_count);
							}
							if (_canceled) {
								pcap_dump_close(dumpfile);
								pcap_close(input_handle);
								std::remove(file_path.c_str());
								_canceled = false;
								return;
							}
						}
					}
				}

				pcap_close(input_handle);

			} else { // skip pcap if it doesn't contain any selected packets
				packet_index += pcap_packets_count;
			}

			start += pcap_packets_count;
		}

		pcap_dump_close(dumpfile);

		if (_f_progress) {
			_f_progress(exported_packets_count);
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

  protected:
	const PVRush::PVInputType::list_inputs& _inputs;
	PVRush::PVNraw const& _nraw;
	std::vector<std::pair<std::string, size_t>> _input_pcaps;
	pvcop::db::indexes _sorted_indexes;
	bool _export_complete_stream = true;
	bool _open_pcap_after_export = true;
};

class PVPcapNgExporter : public PVPcapExporter
{
  public:
	PVPcapNgExporter(const PVRush::PVInputType::list_inputs& inputs, PVRush::PVNraw const& nraw)
	    : PVPcapExporter(inputs, nraw)
	{
	}

  public:
	void export_pcap(const std::string& file_path, const PVCore::PVSelBitField& sel) override
	{
		pvcop::core::array<pvcop::db::index_t> sort_order = _sorted_indexes.to_core_array();

		// output pcap
		light_pcapng_t* dumpfile =
		    light_pcapng_open_write(file_path.c_str(), light_create_default_file_info());

		light_packet_header pkt_header;
		const uint8_t* pkt_data = NULL;

		size_t packet_index = 0;
		size_t exported_packets_count = 0;
		size_t start = 0;
		size_t end = 0;

		for (const std::pair<std::string, size_t>& p : _input_pcaps) {

			const std::string& pcap_path = p.first;
			size_t pcap_packets_count = p.second;

			end += pcap_packets_count;

			if (sel.bit_count(start, end - 1) != 0) {

				light_pcapng_t* input_handle =
				    light_pcapng_open_read(pcap_path.c_str(), LIGHT_TRUE);

				while (light_get_next_packet(input_handle, &pkt_header, &pkt_data)) {
					if (pkt_data != NULL and sel.get_line(sort_order[packet_index++])) {
						light_write_packet(dumpfile, &pkt_header, pkt_data);

						if (exported_packets_count++ % PACKETS_STEP_COUNT == 0) {
							if (_f_progress) {
								_f_progress(exported_packets_count);
							}
							if (_canceled) {
								light_pcapng_close(dumpfile);
								light_pcapng_close(input_handle);
								std::remove(file_path.c_str());
								_canceled = false;
								return;
							}
						}
					}
				}

				light_pcapng_close(input_handle);

			} else { // skip pcap if it doesn't contain any selected packets
				packet_index += pcap_packets_count;
			}

			start += pcap_packets_count;
		}

		light_pcapng_close(dumpfile);

		_f_progress(exported_packets_count);
	}
};

} // namespace PVCore

#endif // __PVCORE_PVEXPORTER_H__
