/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVConfig.h>

#include <pvkernel/filter/PVChunkFilterByElt.h>

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
    : PVSource(scene,
               inputs,
               sc,
               format,
               0,
               PVCore::PVConfig::get()
                   .config()
                   .value("pvkernel/extract_first", PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT)
                   .toInt())
{
}

PVRush::PVFormat& populated_format(PVRush::PVFormat& format)
{
	format.populate();
	return format;
}

Inendi::PVSource::PVSource(Inendi::PVScene& scene,
                           PVRush::PVInputType::list_inputs const& inputs,
                           PVRush::PVSourceCreator_p sc,
                           PVRush::PVFormat const& format,
                           size_t ext_start,
                           size_t ext_end)
    : PVCore::PVDataTreeChild<PVScene, PVSource>(scene)
    , _format(format)
    , _nraw()
    , _extractor(populated_format(_format), _nraw)
    , _inputs(inputs)
    , _src_plugin(sc)
{

	if (inputs.empty()) {
		throw PVRush::PVInputException("Source can't be created without input");
	}

	QSettings& pvconfig = PVCore::PVConfig::get().config();

	// Set extractor default values
	_extractor.set_last_start(ext_start);
	_extractor.set_last_nlines(ext_end);

	int nchunks = pvconfig.value("pvkernel/number_living_chunks", 0).toInt();
	if (nchunks != 0) {
		_extractor.set_number_living_chunks(nchunks);
	}

	// Set sources
	files_append_noextract();
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

void Inendi::PVSource::files_append_noextract()
{
	for (int i = 0; i < _inputs.count(); i++) {
		PVRush::PVSourceCreator::source_p src =
		    _src_plugin->create_source_from_input(_inputs[i], _format);
		_extractor.add_source(src);
	}
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
	extract_finished();
}

void Inendi::PVSource::load_from_disk(std::string const& nraw_folder)
{
	_nraw.load_from_disk(nraw_folder);
}

void Inendi::PVSource::extract_finished()
{
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

PVCol Inendi::PVSource::get_column_count() const
{
	return _format.get_axes().size();
}

std::string Inendi::PVSource::get_value(PVRow row, PVCol col) const
{
	assert(row < get_row_count());
	assert(col < get_column_count());

	return _nraw.at_string(row, col);
}

std::string Inendi::PVSource::get_input_value(PVRow row, PVCol col, bool* res) const
{
	assert(row < get_row_count());
	assert(col < get_column_count());

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
	assert(col < get_column_count());

	const PVRush::PVNraw::unconvertable_values_t& unconv = get_rushnraw().unconvertable_values();

	return unconv.has_failed(row, col);
}

QString Inendi::PVSource::get_window_name() const
{
	const size_t line_start = get_extraction_last_start();
	const size_t line_end = line_start + get_row_count() - 1;
	return QString::fromStdString(get_name()) + QString(" / ") + get_format_name() +
	       QString("\n(%L1 -> %L2)").arg(line_start).arg(line_end);
}

QString Inendi::PVSource::get_tooltip() const
{
	const size_t line_start = get_extraction_last_start();
	const size_t line_end = line_start + get_row_count() - 1;

	QString source = QString("source: %1").arg(QString::fromStdString(get_name()));
	QString format = QString("format: %1").arg(get_format_name());
	QString range = QString("range: %L1 - %L2").arg(line_start).arg(line_end);

	return source + "\n" + format + "\n" + range;
}

void Inendi::PVSource::serialize_write(PVCore::PVSerializeObject& so)
{
	QString src_name = _src_plugin->registered_name();
	so.attribute("source-plugin", src_name);

	// Save the state of the extractor
	chunk_index start, nlines;
	start = _extractor.get_last_start();
	nlines = _extractor.get_last_nlines();
	so.attribute("index_start", start);
	so.attribute("nlines", nlines);

	QString nraw_path = QString::fromStdString(get_rushnraw().collection().rootdir());

	// Save the format
	so.object("format", _format, QObject::tr("Format"));

	// Serialize Input description to reload data if required.
	QString type_name = _src_plugin->supported_type();
	so.attribute("source-type", type_name);

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
}

Inendi::PVSource& Inendi::PVSource::serialize_read(PVCore::PVSerializeObject& so, PVScene& parent)
{
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
	int idx = 0;
	try {
		while (true) {
			// FIXME It throws when there are no more data collections.
			// It should not be an exception as it is a normal behavior.
			PVCore::PVSerializeObject_p new_obj =
			    list_inputs->create_object(QString::number(idx++));
			inputs_for_type.push_back(int_lib->serialize_read(*new_obj));
		}
	} catch (PVCore::PVSerializeArchiveErrorNoObject const&) {
	}

	QString src_name;
	so.attribute("source-plugin", src_name);
	// FIXME : Handle error when source name if not correct
	PVRush::PVSourceCreator_p sc_lib =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name(src_name);

	PVRush::PVFormat format;
	so.object("format", format);

	// Get the state of the extractor
	chunk_index start, nlines;
	so.attribute("index_start", start);
	so.attribute("nlines", nlines);

	PVSource& source = parent.emplace_add_child(inputs_for_type, sc_lib, format, start, nlines);

	QString nraw_folder;
	so.attribute("nraw_path", nraw_folder, QString());

	if (not nraw_folder.isEmpty()) {
		QString user_based_nraw_dir = PVRush::PVNrawCacheManager::nraw_dir() + QDir::separator() +
		                              QDir(nraw_folder).dirName();
		QFileInfo fi(user_based_nraw_dir);
		if (fi.exists() and fi.isDir()) {
			nraw_folder = user_based_nraw_dir;
		} else {
			nraw_folder = QString();
		}
	}

	source.load_data(nraw_folder.toStdString());

	// Create the list of mapped
	PVCore::PVSerializeObject_p list_obj = so.create_object(
	    source.get_children_serialize_name(), source.get_children_description(), true, true);
	idx = 0;
	try {
		while (true) {
			// FIXME It throws when there are no more data collections.
			// It should not be an exception as it is a normal behavior.
			PVCore::PVSerializeObject_p new_obj = list_obj->create_object(QString::number(idx++));
			PVMapped::serialize_read(*new_obj, source);
		}
	} catch (PVCore::PVSerializeArchiveErrorNoObject const&) {
	}

	return source;
}
