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

#ifndef __PVPCAPDESCRIPTION_H__
#define __PVPCAPDESCRIPTION_H__

#include <pvkernel/rush/PVFileDescription.h>

#include "libpvpcap/include/libpvpcap/extraction.h"
#include "libpvpcap/include/libpvpcap/ws.h"

#include <QStringList>

#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace PVRush
{

/**
 * What the captures imported together share: the frames and the conversations of each, which
 * the next one numbers its own after.
 *
 * Thread-safety: may be used from any thread.
 */
class PVPcapImport
{
  public:
	explicit PVPcapImport(size_t captures) : _counts(captures) {}

	//! Frames and conversations of the captures before `index`, if they were all read.
	std::optional<std::pair<uint64_t, uint64_t>> before(size_t index) const
	{
		const std::lock_guard<std::mutex> lock(_mutex);
		std::pair<uint64_t, uint64_t> total{0, 0};
		for (size_t i = 0; i < index; i++) {
			if (not _counts[i]) {
				return std::nullopt;
			}
			total.first += _counts[i]->first;
			total.second += _counts[i]->second;
		}
		return total;
	}

	void read(size_t index, uint64_t frames, uint64_t streams)
	{
		const std::lock_guard<std::mutex> lock(_mutex);
		_counts[index] = std::make_pair(frames, streams);
	}

  private:
	mutable std::mutex _mutex;
	std::vector<std::optional<std::pair<uint64_t, uint64_t>>> _counts;
};

/**
 * A capture to import, and what tshark is asked of it.
 *
 * Its frames are numbered after those of the captures imported before it, and so are its
 * conversations: frame.global_number and stream_id are unique among all of them.
 */
class PVPcapDescription : public PVFileDescription
{
  public:
	PVPcapDescription(QString const& capture,
	                  pvpcap::extraction::options extraction,
	                  std::shared_ptr<PVPcapImport> import,
	                  size_t index,
	                  bool multi_inputs)
	    : PVFileDescription(capture, multi_inputs)
	    , _extraction(std::move(extraction))
	    , _import(std::move(import))
	    , _index(index)
	{
	}

	//! A capture whose offsets were recorded rather than shared.
	PVPcapDescription(QString const& capture,
	                  pvpcap::extraction::options extraction,
	                  uint64_t frames_offset,
	                  uint64_t streams_offset,
	                  bool multi_inputs)
	    : PVFileDescription(capture, multi_inputs)
	    , _extraction(std::move(extraction))
	    , _frames_offset(frames_offset)
	    , _streams_offset(streams_offset)
	{
	}

	QString original_pcap_path() const { return path(); }

	const pvpcap::extraction::options& extraction() const { return _extraction; }

	/**
	 * Whether what tshark is asked is known: an investigation saved before it was recorded
	 * can't have its capture read again.
	 */
	bool can_extract() const { return not _extraction.fields.empty(); }

	/**
	 * The frames and the conversations of the captures imported before this one.
	 *
	 * @return nothing as long as one of those was not read
	 */
	std::optional<std::pair<uint64_t, uint64_t>> offsets() const
	{
		if (_import) {
			return _import->before(_index);
		}
		return std::make_pair(_frames_offset, _streams_offset);
	}

	//! Tells the captures imported after this one how many frames and conversations it has.
	void read(uint64_t frames, uint64_t streams)
	{
		_frames = frames;
		if (_import) {
			_import->read(_index, frames, streams);
		}
	}

	//! Number of frames of the capture, once it was read.
	uint64_t frames() const { return _frames; }

  public:
	void serialize_write(PVCore::PVSerializeObject& so) const override
	{
		so.set_current_status("Saving source file information...");

		if (so.save_log_file()) {
			QFileInfo fi(_path);
			QString fname = fi.fileName();
			so.file_write(fname, _path);
			so.attribute_write("file_path", fname);
		} else {
			so.attribute_write("file_path", _path);
		}
		so.attribute_write("multi_inputs", _multi_inputs);
		const auto known = offsets();
		so.attribute_write("packets_count_offset", qulonglong(known ? known->first : 0));
		so.attribute_write("streams_offset", qulonglong(known ? known->second : 0));

		std::vector<QString> options;
		for (const std::string& option : _extraction.tshark_options) {
			options.push_back(QString::fromStdString(option));
		}
		so.list_attributes("tshark_options", options);
		std::vector<QString> names;
		std::vector<int> kinds;
		for (const auto& field : _extraction.fields) {
			names.push_back(QString::fromStdString(field.name));
			kinds.push_back(static_cast<int>(field.type));
		}
		so.list_attributes("field_names", names);
		so.list_attributes("field_kinds", kinds);
		so.attribute_write("aggregator", QString::fromStdString(_extraction.aggregator));
		so.attribute_write("two_pass", _extraction.two_pass);
	}

	static std::unique_ptr<PVInputDescription> serialize_read(PVCore::PVSerializeObject& so)
	{
		so.set_current_status("Loading source file information...");
		QString path = so.attribute_read<QString>("file_path");

		if (not QFileInfo(path).isReadable()) {
			try {
				path = so.file_read(path);
			} catch (PVCore::PVSerializeArchiveError const&) {
				if (so.is_repaired_error()) {
					path = QString::fromStdString(so.get_repaired_value());
				} else {
					throw PVCore::PVSerializeReparaibleFileError(
					    "Source file: '" + path.toStdString() + "' can't be found",
					    so.get_logical_path().toStdString(), path.toStdString());
				}
			}
		}

		// Investigations saved before what tshark is asked was recorded have none of it,
		// and can't do without their NRAW.
		pvpcap::extraction::options extraction;
		std::vector<QString> options;
		std::vector<QString> names;
		std::vector<int> kinds;
		so.list_attributes("tshark_options", options);
		so.list_attributes("field_names", names);
		so.list_attributes("field_kinds", kinds);
		for (const QString& option : options) {
			extraction.tshark_options.push_back(option.toStdString());
		}
		for (size_t i = 0; i < names.size() and i < kinds.size(); i++) {
			extraction.fields.push_back(
			    {names[i].toStdString(), static_cast<pvpcap::extraction::field::kind>(kinds[i])});
		}
		const QString aggregator = so.attribute_read<QString>("aggregator");
		if (not aggregator.isEmpty()) {
			extraction.aggregator = aggregator.toStdString();
		}
		extraction.two_pass = so.attribute_read<bool>("two_pass");
		// Of the machine reading the capture, not of the one that saved the investigation.
		extraction.environment = pvpcap::tshark_environment();

		const uint64_t frames_offset = so.attribute_read<qulonglong>("packets_count_offset");
		const uint64_t streams_offset = so.attribute_read<qulonglong>("streams_offset");
		const bool multi_inputs = so.attribute_read<bool>("multi_inputs");
		return std::unique_ptr<PVInputDescription>(new PVPcapDescription(
		    path, std::move(extraction), frames_offset, streams_offset, multi_inputs));
	}

  private:
	pvpcap::extraction::options _extraction;
	std::shared_ptr<PVPcapImport> _import; //!< null once deserialized: the offsets are below
	size_t _index = 0;
	uint64_t _frames_offset = 0;
	uint64_t _streams_offset = 0;
	uint64_t _frames = 0;
};

} // namespace PVRush

#endif // __PVPCAPDESCRIPTION_H__
