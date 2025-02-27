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

#include <pvkernel/core/PVConfig.h>

#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include <squey/PVScene.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>
#include <squey/PVRoot.h>

#include <QCryptographicHash>

Squey::PVSource::PVSource(Squey::PVScene& scene,
                           PVRush::PVInputType::list_inputs const& inputs,
                           PVRush::PVSourceCreator_p sc,
                           PVRush::PVFormat const& format)
    : PVCore::PVDataTreeChild<PVScene, PVSource>(scene)
    , _original_format(format)
    , _format((not sc->custom_multi_inputs() and inputs.size() > 1)
                  ? _original_format.add_input_name_column()
                  : format)
    , _nraw()
    , _inputs(inputs)
    , _output(_nraw)
    , _src_plugin(sc)
    , _extractor(get_format(), _output, _src_plugin, _inputs)
{
	if (inputs.empty()) {
		throw PVRush::PVInputException("Source can't be created without input");
	}
}

Squey::PVSource::~PVSource()
{
	get_parent<PVRoot>().source_being_deleted(this);
	PVLOG_DEBUG("In PVSource destructor: %p\n", this);
}

Squey::PVView* Squey::PVSource::current_view()
{
	PVView* view = get_parent<PVRoot>().current_view();
	if (&view->get_parent<PVSource>() == this) {
		return view;
	}
	return nullptr;
}

Squey::PVView const* Squey::PVSource::current_view() const
{
	PVView const* view = get_parent<PVRoot>().current_view();
	if (&view->get_parent<PVSource>() == this) {
		return view;
	}
	return nullptr;
}

PVRush::PVControllerJob_p Squey::PVSource::extract(size_t skip_lines_count)
{
	PVRush::PVControllerJob_p job = _extractor.process_from_agg_nlines(skip_lines_count);

	return job;
}

void Squey::PVSource::wait_extract_end(PVRush::PVControllerJob_p job)
{
	job->wait_end();
	_inv_elts = job->get_invalid_evts();
	_extractor.release_inputs(not job->running());
}

PVRush::PVNraw& Squey::PVSource::get_rushnraw()
{
	return _nraw;
}

const PVRush::PVNraw& Squey::PVSource::get_rushnraw() const
{
	return _nraw;
}

PVRow Squey::PVSource::get_row_count() const
{
	return _nraw.row_count();
}

PVRow Squey::PVSource::get_valid_row_count() const
{
	return _nraw.get_valid_row_count();
}

PVCol Squey::PVSource::get_nraw_column_count() const
{
	return PVCol(get_format().get_axes().size());
}

std::string Squey::PVSource::get_value(PVRow row, PVCol col) const
{
	assert(row < get_row_count());
	assert(col < get_nraw_column_count());

	return _nraw.at_string(row, col);
}

bool Squey::PVSource::is_valid(PVRow row, PVCol col) const
{
	assert(row < get_row_count());
	assert(col < get_nraw_column_count());

	return get_rushnraw().column(col).is_valid(row);
}

pvcop::db::INVALID_TYPE Squey::PVSource::has_invalid(PVCol col) const
{
	assert(col < get_nraw_column_count());

	return get_rushnraw().column(col).has_invalid();
}

QString Squey::PVSource::get_window_name() const
{
	return QString::fromStdString(get_name()) + QString(" / ") + get_format_name();
}

QString Squey::PVSource::get_tooltip() const
{
	QString source = QString("source: %1").arg(QString::fromStdString(get_name()));
	QString format = QString("format: %1").arg(get_format_name());

	return source + "\n" + format;
}

std::string Squey::PVSource::hash() const
{
	QCryptographicHash hasher(QCryptographicHash::Md5);
	constexpr size_t max_line_hash =
	    500000; // Just use a subset. It is not optimal but it have to be "fast enough"
	for (size_t j = 0; j < std::min<size_t>(_nraw.row_count(), max_line_hash); j++) {
		std::string r = get_value(j, PVCol(0));
		hasher.addData(QString::fromStdString(r).toUtf8());
	}
	std::string size = std::to_string(_nraw.row_count());
	hasher.addData(QString::fromStdString(size).toUtf8());
	return hasher.result().data();
}

void Squey::PVSource::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.set_current_status("Saving source...");
	QString src_name = _src_plugin->registered_name();
	so.attribute_write("source-plugin", src_name);

	PVCore::PVSerializeObject_p nraw_obj = so.create_object("nraw");
	_nraw.serialize_write(*nraw_obj);

	so.set_current_status("Computing raw data integrity...");
	std::string h = hash();
	so.buffer_write("src_hash", (char*)h.data(), 16);

	// Save the format
	so.set_current_status("Saving format...");
	PVCore::PVSerializeObject_p format_obj = so.create_object("format");
	_original_format.serialize_write(*format_obj);

	// Serialize Input description to reload data if required.
	QString type_name = _src_plugin->supported_type();
	so.attribute_write("source-type", type_name);

	so.set_current_status("Saving inputs...");
	PVCore::PVSerializeObject_p list_inputs = so.create_object("inputs");
	int idx = 0;
	for (PVRush::PVInputDescription_p const& input : _inputs) {
		PVCore::PVSerializeObject_p new_input = list_inputs->create_object(QString::number(idx++));
		input->serialize_write(*new_input);
	}
	so.attribute_write("input_count", idx);

	so.set_current_status("Saving invalid elements information...");
	// Serialize invalid elements.
	int inv_elts_count = _inv_elts.size();
	so.attribute_write("inv_elts_count", inv_elts_count);
	idx = 0;
	for (auto const& inv_elt : _inv_elts) {
		int inv_line = inv_elt.first;
		so.attribute_write(QString::fromStdString("inv_elts_id/" + std::to_string(idx)), inv_line);
		QString inv_content = QString::fromStdString(inv_elt.second);
		so.attribute_write(QString::fromStdString("inv_elts_value/" + std::to_string(idx)),
		                   inv_content);
		idx++;
	}

	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj = so.create_object("mapped");
	idx = 0;
	for (PVMapped const* mapped : get_children()) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx++));
		mapped->serialize_write(*new_obj);
	}
	so.attribute_write("mapped_count", idx);
}

Squey::PVSource& Squey::PVSource::serialize_read(PVCore::PVSerializeObject& so, PVScene& parent)
{
	so.set_current_status("Loading source...");
	// Reload input desription
	auto type_name = so.attribute_read<QString>("source-type");
	// FIXME : We should check for type_name validity if archive was manually changed.
	PVRush::PVInputType_p int_lib =
	    LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name(type_name);

	// Get the inputs list object for that input type
	PVRush::PVInputType::list_inputs_desc inputs_for_type;

	// Create the list of input
	PVCore::PVSerializeObject_p list_inputs = so.create_object("inputs");
	int input_count = so.attribute_read<int>("input_count");
	for (int idx = 0; idx < input_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_inputs->create_object(QString::number(idx));
		inputs_for_type.push_back(int_lib->serialize_read(*new_obj));
	}

	auto src_name = so.attribute_read<QString>("source-plugin");
	// FIXME : Handle error when source name if not correct
	PVRush::PVSourceCreator_p sc_lib =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name(src_name);

	// read Format
	PVCore::PVSerializeObject_p format_obj = so.create_object("format");
	PVRush::PVFormat format = PVRush::PVFormat::serialize_read(*format_obj);

	PVSource& source = parent.emplace_add_child(inputs_for_type, sc_lib, format);

	try {
		so.set_current_status("Loading raw data...");
		PVCore::PVSerializeObject_p nraw_obj = so.create_object("nraw");
		source._nraw = PVRush::PVNraw::serialize_read(*nraw_obj);

		// Serialize invalid elements.
		so.set_current_status("Loading invalid events information...");
		int inv_elts_count = so.attribute_read<int>("inv_elts_count");
		for (int idx = 0; idx < inv_elts_count; idx++) {
			int inv_line = so.attribute_read<int>(
			    QString::fromStdString("inv_elts_id/" + std::to_string(idx)));
			auto inv_content = so.attribute_read<QString>(
			    QString::fromStdString("inv_elts_value/" + std::to_string(idx)));
			source._inv_elts.emplace(inv_line, inv_content.toStdString());
		}
	} catch (PVRush::NrawLoadingFail const& e) {
		so.set_current_status("No raw data in cache, reloading it from original source file...");
		source.load_data();

		std::string hash_value(16, ' ');
		so.buffer_read("src_hash", (char*)hash_value.data(), 16);
		so.set_current_status("Checking raw data integrity...");
		if (source.hash() != hash_value) {
			throw PVCore::PVSerializeArchiveError("Source mismatch with the saved one.");
		}
	}

	// Create the list of mapped
	PVCore::PVSerializeObject_p list_obj = so.create_object("mapped");
	int mapped_count = so.attribute_read<int>("mapped_count");
	for (int idx = 0; idx < mapped_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
		PVMapped::serialize_read(*new_obj, source);
	}

	return source;
}
