/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2017
 */
#ifndef __PVPCAPDESCRIPTION_H__
#define __PVPCAPDESCRIPTION_H__

#include <pvkernel/rush/PVFileDescription.h>
#include "libpvpcap/include/libpvpcap/pcap_splitter.h"

namespace PVRush
{

class PVPcapDescription : public PVFileDescription
{
  public:
	PVPcapDescription(QString const& csv_path,
	                  QString const& original_pcap_path,
	                  size_t pcap_packets_count_offset,
	                  pvpcap::splitted_file_t::vector_uint32_sp pi,
	                  size_t pcap_streams_id_offset,
	                  pvpcap::splitted_file_t::vector_uint32_sp si,
	                  bool multi_inputs)
	    : PVFileDescription(csv_path)
	    , _original_pcap_path(original_pcap_path)
	    , _pcap_packets_count_offset(pcap_packets_count_offset)
	    , _packets_indexes(pi)
	    , _pcap_streams_id_offset(pcap_streams_id_offset)
	    , _streams_ids(si)
	    , _multi_inputs(multi_inputs)
	{
	}

	QString original_pcap_path() const { return _original_pcap_path; }
	size_t pcap_packets_count_offset() const { return _pcap_packets_count_offset; }
	size_t pcap_streams_id_offset() const { return _pcap_streams_id_offset; }
	bool multi_inputs() const { return _multi_inputs; }
	const pvpcap::splitted_file_t::vector_uint32_t& packets_indexes() const
	{
		return *_packets_indexes.get();
	}
	void clear_packets_indexes() { _packets_indexes->clear(); }
	const pvpcap::splitted_file_t::vector_uint32_t& streams_ids() const
	{
		return *_streams_ids.get();
	}
	void reset_streams_ids() { _streams_ids.reset(); }

  public:
	void serialize_write(PVCore::PVSerializeObject& so) const override
	{
		so.set_current_status("Saving source file information...");

		if (so.save_log_file()) {
			QFileInfo fi(_original_pcap_path);
			QString fname = fi.fileName();
			so.file_write(fname, _original_pcap_path);
			// FIXME : Before, we still reload data from original file, never from packed file.
			so.attribute_write("file_path", fname);
		} else {
			so.attribute_write("file_path", _original_pcap_path);
		}
	}

	static std::unique_ptr<PVInputDescription> serialize_read(PVCore::PVSerializeObject& so)
	{
		so.set_current_status("Loading source file information...");
		QString path = so.attribute_read<QString>("file_path");

		// File exists, continue with it
		if (QFileInfo(path).isReadable()) {
			// FIXME : As long as the pcap preprocessing phase is not properly integrated to
			//         the import pipeline, it will not be possible to reload an investigation
			//         with a missing cache (ie. from the original pcap files)
			return std::unique_ptr<PVInputDescription>(
			    new PVPcapDescription("", path, 0, {}, 0, {}, false));
		} else {
			throw PVCore::PVSerializeReparaibleFileError(
			    "Source file: '" + path.toStdString() + "' can't be found",
			    so.get_logical_path().toStdString(), path.toStdString());
		}
	}

  private:
	QString _original_pcap_path;
	size_t _pcap_packets_count_offset;
	pvpcap::splitted_file_t::vector_uint32_sp _packets_indexes;
	size_t _pcap_streams_id_offset;
	pvpcap::splitted_file_t::vector_uint32_sp _streams_ids;
	bool _multi_inputs;
};

} // namespace PVRush

#endif // __PVPCAPDESCRIPTION_H__
