/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVConfig.h>

#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVNrawOutput.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include <inendi/PVScene.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>
#include <inendi/PVRoot.h>

Inendi::PVSource::PVSource(Inendi::PVScene& scene,
                           PVRush::PVInputType::list_inputs const& inputs,
                           PVRush::PVSourceCreator_p sc,
                           PVRush::PVFormat const& format)
    : PVCore::PVDataTreeChild<PVScene, PVSource>(scene)
    , _format(format)
    , _nraw()
    , _inputs(inputs)
    , _src_plugin(sc)
    , _extractor(_format, _nraw, _src_plugin, _inputs)
{
	if (inputs.empty()) {
		throw PVRush::PVInputException("Source can't be created without input");
	}
}

Inendi::PVSource::~PVSource()
{
	get_parent<PVRoot>().source_being_deleted(this);
	PVLOG_DEBUG("In PVSource destructor: %p\n", this);
}

Inendi::PVView* Inendi::PVSource::current_view()
{
	PVView* view = get_parent<PVRoot>().current_view();
	if (&view->get_parent<PVSource>() == this) {
		return view;
	}
	return nullptr;
}

Inendi::PVView const* Inendi::PVSource::current_view() const
{
	PVView const* view = get_parent<PVRoot>().current_view();
	if (&view->get_parent<PVSource>() == this) {
		return view;
	}
	return nullptr;
}

PVRush::PVControllerJob_p Inendi::PVSource::extract(size_t skip_lines_count)
{
	PVRush::PVControllerJob_p job = _extractor.process_from_agg_nlines(skip_lines_count);

	return job;
}

void Inendi::PVSource::wait_extract_end(PVRush::PVControllerJob_p job)
{
	job->wait_end();
	_inv_elts = job->get_invalid_evts();
	_extractor.release_inputs();
}

PVRush::PVNraw& Inendi::PVSource::get_rushnraw()
{
	return _nraw;
}

const PVRush::PVNraw& Inendi::PVSource::get_rushnraw() const
{
	return _nraw;
}

PVRow Inendi::PVSource::get_row_count() const
{
	return _nraw.get_row_count();
}

PVRow Inendi::PVSource::get_valid_row_count() const
{
	return _nraw.get_valid_row_count();
}

PVCol Inendi::PVSource::get_nraw_column_count() const
{
	return _format.get_axes().size();
}

std::string Inendi::PVSource::get_value(PVRow row, PVCol col) const
{
	assert(row < get_row_count());
	assert(col < get_nraw_column_count());

	return _nraw.at_string(row, col);
}

std::string Inendi::PVSource::get_input_value(PVRow row, PVCol col, bool* res) const
{
	assert(row < get_row_count());
	assert(col < get_nraw_column_count());

	const PVRush::PVNraw::unconvertable_values_t& unconv = get_rushnraw().unconvertable_values();

	bool conversion_failed;
	std::string str = unconv.get(row, col, &conversion_failed);

	if (res) {
		*res = conversion_failed;
	}

	if (conversion_failed) {
		return str;
	} else {
		return get_value(row, col);
	}
}

bool Inendi::PVSource::has_conversion_failed(PVRow row, PVCol col) const
{
	assert(row < get_row_count());
	assert(col < get_nraw_column_count());

	const PVRush::PVNraw::unconvertable_values_t& unconv = get_rushnraw().unconvertable_values();

	return unconv.has_failed(row, col);
}

QString Inendi::PVSource::get_window_name() const
{
	return QString::fromStdString(get_name()) + QString(" / ") + get_format_name();
}

QString Inendi::PVSource::get_tooltip() const
{
	QString source = QString("source: %1").arg(QString::fromStdString(get_name()));
	QString format = QString("format: %1").arg(get_format_name());

	return source + "\n" + format;
}

void Inendi::PVSource::serialize_write(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Serialize source.");
	QString src_name = _src_plugin->registered_name();
	so.attribute("source-plugin", src_name);

	PVCore::PVSerializeObject_p nraw_obj = so.create_object("nraw", "NRaw", true, true);
	_nraw.serialize_write(*nraw_obj);

	// Save the format
	so.set_current_status("Serialize Format.");
	PVCore::PVSerializeObject_p format_obj = so.create_object("format", "Format", true, true);
	_format.serialize_write(*format_obj);

	// Serialize Input description to reload data if required.
	QString type_name = _src_plugin->supported_type();
	so.attribute("source-type", type_name);

	so.set_current_status("Serialize Inputs.");
	PVCore::PVSerializeObject_p list_inputs =
	    so.create_object("inputs", "Description of inputs", true, true);
	int idx = 0;
	for (PVRush::PVInputDescription_p const& input : _inputs) {
		QString child_name = QString::number(idx++);
		PVCore::PVSerializeObject_p new_input = list_inputs->create_object(
		    child_name, _src_plugin->supported_type_lib()->human_name_of_input(input), false);
		input->serialize_write(*new_input);
		new_input->set_bound_obj(*input);
	}
	so.attribute("input_count", idx);

	so.set_current_status("Serialize invalid elements.");
	// Serialize invalid elements.
	int inv_elts_count = _inv_elts.size();
	so.attribute("inv_elts_count", inv_elts_count);
	idx = 0;
	for (auto const& inv_elt : _inv_elts) {
		int inv_line = inv_elt.first;
		so.attribute(QString::fromStdString("inv_elts_id/" + std::to_string(idx)), inv_line);
		QString inv_content = QString::fromStdString(inv_elt.second);
		so.attribute(QString::fromStdString("inv_elts_value/" + std::to_string(idx)), inv_content);
		idx++;
	}

	// Read the data colletions
	PVCore::PVSerializeObject_p list_obj =
	    so.create_object(get_children_serialize_name(), get_children_description(), true, true);
	idx = 0;
	for (PVMapped* mapped : get_children()) {
		QString child_name = QString::number(idx++);
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(
		    child_name, QString::fromStdString(mapped->get_serialize_description()), false);
		mapped->serialize_write(*new_obj);
		new_obj->set_bound_obj(*mapped);
	}
	so.attribute("mapped_count", idx);
}

Inendi::PVSource& Inendi::PVSource::serialize_read(PVCore::PVSerializeObject& so, PVScene& parent)
{
	so.set_current_status("Loading Source");
	// Reload input desription
	QString type_name;
	so.attribute("source-type", type_name);
	// FIXME : We should check for type_name validity if archive was manually changed.
	PVRush::PVInputType_p int_lib =
	    LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name(type_name);

	// Get the inputs list object for that input type
	PVRush::PVInputType::list_inputs_desc inputs_for_type;

	// Create the list of input
	PVCore::PVSerializeObject_p list_inputs =
	    so.create_object("inputs", "Description of inputs", true, true);
	int input_count;
	so.attribute("input_count", input_count);
	for (int idx = 0; idx < input_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_inputs->create_object(QString::number(idx));
		inputs_for_type.push_back(int_lib->serialize_read(*new_obj));
	}

	QString src_name;
	so.attribute("source-plugin", src_name);
	// FIXME : Handle error when source name if not correct
	PVRush::PVSourceCreator_p sc_lib =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name(src_name);

	// read Format
	PVCore::PVSerializeObject_p format_obj = so.create_object("format", "Format", true, true);
	PVRush::PVFormat format = PVRush::PVFormat::serialize_read(*format_obj);

	PVSource& source = parent.emplace_add_child(inputs_for_type, sc_lib, format);

	try {
		so.set_current_status("Load NRaw");
		PVCore::PVSerializeObject_p nraw_obj = so.create_object("nraw", "NRaw", true, true);
		source._nraw = PVRush::PVNraw::serialize_read(*nraw_obj);

		// Serialize invalid elements.
		so.set_current_status("Load invalid events");
		int inv_elts_count;
		so.attribute("inv_elts_count", inv_elts_count);
		for (int idx = 0; idx < inv_elts_count; idx++) {
			int inv_line;
			so.attribute(QString::fromStdString("inv_elts_id/" + std::to_string(idx)), inv_line);
			QString inv_content;
			so.attribute(QString::fromStdString("inv_elts_value/" + std::to_string(idx)),
			             inv_content);
			source._inv_elts.emplace(inv_line, inv_content.toStdString());
		}
	} catch (PVRush::NrawLoadingFail const& e) {
		so.set_current_status("Fail to load NRaw from cache, reload it from source file");
		source.load_data();
	}

	// Create the list of mapped
	PVCore::PVSerializeObject_p list_obj = so.create_object(
	    source.get_children_serialize_name(), source.get_children_description(), true, true);
	int mapped_count;
	so.attribute("mapped_count", mapped_count);
	for (int idx = 0; idx < mapped_count; idx++) {
		PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx));
		PVMapped::serialize_read(*new_obj, source);
	}

	return source;
}
