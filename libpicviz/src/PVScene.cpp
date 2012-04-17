//! \file PVScene.cpp
//! $Id: PVScene.cpp 2875 2011-05-19 04:18:05Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <pvkernel/core/PVSerializeArchiveOptions.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <picviz/PVRoot.h>
#include <picviz/PVScene.h>
#include <picviz/PVSource.h>


#define ARCHIVE_SCENE_DESC (QObject::tr("Workspace"))
/******************************************************************************
 *
 * Picviz::PVScene::PVScene
 *
 *****************************************************************************/
Picviz::PVScene::PVScene(QString scene_name, PVRoot* parent):
	_root(parent),
	_name(scene_name)
{
}

/******************************************************************************
 *
 * Picviz::PVScene::~PVScene
 *
 *****************************************************************************/
Picviz::PVScene::~PVScene()
{
	PVLOG_INFO("In PVScene destructor\n");
}

void Picviz::PVScene::add_source(PVSource_p src)
{
	// For information, from PVScene.h:
	// typedef std::map<PVRush::PVInputType, std::pair<list_sources_t, PVRush::PVInputType::list_inputs> > hash_type_sources_t;
	// hash_type_sources_t _sources;
	
	src->set_parent(this);
	std::pair<list_sources_t, PVRush::PVInputType::list_inputs>& type_srcs = _sources[*(src->get_input_type())];

	PVRush::PVInputType::list_inputs& inputs(type_srcs.second);
	list_sources_t& sources(type_srcs.first);

	// Add sources' inputs to our `inputs' if they do not exist yet
	PVRush::PVInputType::list_inputs src_inputs = src->get_inputs();
	PVRush::PVInputType::list_inputs::const_iterator it;
	for (it = src_inputs.begin(); it != src_inputs.end(); it++) {
		if (!inputs.contains(*it)) {
			inputs.push_back(*it);
		}
	}

	// Add this source to the list of sources for this type
	sources.push_back(src);
}

Picviz::PVScene::list_sources_t Picviz::PVScene::get_all_sources() const
{
	list_sources_t ret;
	hash_type_sources_t::const_iterator it;
	for (it = _sources.begin(); it != _sources.end(); it++) {
		std::pair<list_sources_t, PVRush::PVInputType::list_inputs> const& type_srcs = it->second;
		ret.append(type_srcs.first);
	}
	return ret;
}

Picviz::PVScene::list_views_t Picviz::PVScene::get_all_views() const
{
	list_views_t ret;
	list_sources_t sources = get_all_sources();
	foreach (PVSource_p source, sources) {
		PVSource::list_views_t const& views = source->get_views();
		foreach (Picviz::PVView_p view, views) {
			ret.append(view);
		}
	}

	return ret;
}

Picviz::PVScene::list_sources_t Picviz::PVScene::get_sources(PVRush::PVInputType const& type) const
{
	hash_type_sources_t::const_iterator it = _sources.find(type);
	if (it == _sources.end()) {
		return list_sources_t();
	}
	std::pair<list_sources_t, PVRush::PVInputType::list_inputs> const& type_srcs = it->second;
	return type_srcs.first;
}

bool Picviz::PVScene::del_source(const PVSource* src)
{
	std::pair<list_sources_t, PVRush::PVInputType::list_inputs>& type_srcs = _sources[*(src->get_input_type())];
	list_sources_t& list_srcs(type_srcs.first);

	list_sources_t::iterator it;
	for (it = list_srcs.begin(); it != list_srcs.end(); it++) {
		if (it->get() == src) {
			list_srcs.erase(it);
			return true;
		}
	}

	return false;
}

Picviz::PVRoot* Picviz::PVScene::get_root()
{
	return _root;
}

void Picviz::PVScene::serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	// Get the list of input types
	QStringList input_types;
	so.list_attributes("types", input_types);

	for (int i = 0; i < input_types.size(); i++) {
		// Get the input type lib object
		QString const& type_name = input_types.at(i);
		PVRush::PVInputType_p int_lib = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name(type_name);

		// Get the inputs list object for that input type
		std::pair<list_sources_t, PVRush::PVInputType::list_inputs>& type_srcs = _sources[*int_lib];
		PVRush::PVInputType::list_inputs& inputs(type_srcs.second);
		
		// Get back the inputs
		PVCore::PVSerializeObject_p so_inputs = int_lib->serialize_inputs(so, type_name, inputs);

		// Save the serialize object, so that PVSource objects can refere to them for their inputs
		_so_inputs[*int_lib] = so_inputs;
	}

	// Get the sources
	PVSource_p src(new PVSource());
	src->set_parent(this);
	list_sources_t all_sources;
	so.list("sources", all_sources, QObject::tr("Sources"), src.get());
	src->set_parent(NULL);
	PVLOG_INFO("(PVScene::serialize_read) get %d sources\n", all_sources.size());

	if (!so.has_repairable_errors()) {
		// And finally add them !
		list_sources_t::iterator it;
		for (it = all_sources.begin(); it != all_sources.end(); it++) {
			add_source(*it);
		}
	}
}

void Picviz::PVScene::serialize_write(PVCore::PVSerializeObject& so)
{
	// First serialize the sources.
	// The tree will be like this:
	_so_inputs.clear();
	list_sources_t all_sources;
	hash_type_sources_t::iterator it_type;
	QStringList input_types;
	for (it_type = _sources.begin(); it_type != _sources.end(); it_type++) {
		std::pair<list_sources_t, PVRush::PVInputType::list_inputs>& type_srcs = it_type->second;
		PVRush::PVInputType::list_inputs& inputs(type_srcs.second);
		list_sources_t const& sources(type_srcs.first);

		// Safety check
		if (inputs.size() == 0 || sources.size() == 0) {
			continue;
		}

		// Get the lib object for the input type
		QString type_name = it_type->first.registered_name();
		input_types << type_name;
		PVRush::PVInputType_p int_lib = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name(type_name);
		assert(int_lib);

		PVCore::PVSerializeObject_p so_inputs = int_lib->serialize_inputs(so, int_lib->registered_name(), inputs);
		_so_inputs[it_type->first] = so_inputs;

		// Add these sources to the global list
		all_sources.append(sources);
	}

	so.list_attributes("types", input_types);

	// Then serialize the list of sources
	
	// Get the sources name
	QStringList desc;
	list_sources_t::const_iterator it_src;
	for (it_src = all_sources.begin(); it_src != all_sources.end(); it_src++) {
		desc << (*it_src)->get_name() + QString(" / ") + (*it_src)->get_format_name();
	}
	so.list(QString("sources"), all_sources, QObject::tr("Sources"), (PVSource*) NULL, desc);
}

PVCore::PVSerializeObject_p Picviz::PVScene::get_so_inputs(PVSource const& src)
{
	return _so_inputs[*(src.get_input_type())];
}

PVCore::PVSerializeArchiveOptions_p Picviz::PVScene::get_default_serialize_options()
{
	PVCore::PVSerializeArchiveOptions_p ar(new PVCore::PVSerializeArchiveOptions(PICVIZ_ARCHIVES_VERSION));
	ar->get_root()->object("scene", *this, ARCHIVE_SCENE_DESC);
	return ar;
}

void Picviz::PVScene::save_to_file(QString const& path, PVCore::PVSerializeArchiveOptions_p options, bool save_everything)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchiveZip(path, PVCore::PVSerializeArchive::write, PICVIZ_ARCHIVES_VERSION));
	if (options) {
		ar->set_options(options);
	}
	ar->set_save_everything(save_everything);
	ar->get_root()->object("scene", *this, ARCHIVE_SCENE_DESC);
	ar->finish();
#endif
}

void Picviz::PVScene::load_from_file(QString const& path)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchiveZip(path, PVCore::PVSerializeArchive::read, PICVIZ_ARCHIVES_VERSION));
	load_from_archive(ar);
#endif
}

void Picviz::PVScene::load_from_archive(PVCore::PVSerializeArchive_p ar)
{
#ifdef CUSTOMER_CAPABILITY_SAVE
	ar->get_root()->object("scene", *this, ARCHIVE_SCENE_DESC);
	_original_archive = ar;
#endif
}
