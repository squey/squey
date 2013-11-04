/**
 * \file PVRecentItemsManager.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvkernel/core/PVRecentItemsManager.h>
#include <pvkernel/rush/PVFormat.h>

#define ITEM_SUBKEY_SOURCE_CREATOR_NAME "__sc_name"
#define ITEM_SUBKEY_FORMAT_NAME         "__format_name"
#define ITEM_SUBKEY_FORMAT_PATH         "__format_path"
#define ITEM_SUBKEY_INPUTS              "inputs"

typename PVCore::PVRecentItemsManager::PVRecentItemsManager_p PVCore::PVRecentItemsManager::_recent_items_manager_p = PVRecentItemsManager_p();

void PVCore::PVRecentItemsManager::add(const QString& item_path, Category category)
{
	QString recent_items_key = _recents_items_keys[category];
	QStringList files = _recents_settings.value(recent_items_key).toStringList();
	files.removeAll(item_path);
	files.prepend(item_path);
	if (category != Category::PROJECTS) {
		for (; files.size() > _max_recent_items; files.removeLast()) {}
	}
	_recents_settings.setValue(recent_items_key, files);
	_recents_settings.sync();
}

void PVCore::PVRecentItemsManager::add_source(PVRush::PVSourceCreator_p source_creator_p, const PVRush::PVInputType::list_inputs& inputs, const PVRush::PVFormat& format)
{
	_recents_settings.beginGroup(_recents_items_keys[SOURCES]);

	uint64_t source_timestamp = get_source_timestamp_to_replace(PVRush::PVSourceDescription(inputs, source_creator_p, format));
	if (source_timestamp) {
		_recents_settings.remove(QString::number(source_timestamp));
	}

	_recents_settings.beginGroup(QString::number(QDateTime::currentDateTime().toMSecsSinceEpoch()));

	_recents_settings.setValue(ITEM_SUBKEY_SOURCE_CREATOR_NAME, source_creator_p->registered_name());
	_recents_settings.setValue(ITEM_SUBKEY_FORMAT_NAME, format.get_format_name());
	_recents_settings.setValue(ITEM_SUBKEY_FORMAT_PATH, format.get_full_path());
		_recents_settings.beginWriteArray(ITEM_SUBKEY_INPUTS);
		int inputs_index = 0;
		for (auto input : inputs) {
			_recents_settings.setArrayIndex(inputs_index++);
			input->save_to_qsettings(_recents_settings);
		}
		_recents_settings.endArray();

	_recents_settings.endGroup();

	_recents_settings.endGroup();
	_recents_settings.sync();
}

void PVCore::PVRecentItemsManager::clear(Category category, QList<int> indexes)
{
	if (category == Category::SOURCES) {
		_recents_settings.beginGroup(_recents_items_keys[Category::SOURCES]);
		QStringList sources = _recents_settings.childGroups();
		for (int i = sources.length(); i --> 0; ) {
			_recents_settings.beginGroup(sources.at(i));
			if(indexes.isEmpty() || indexes.contains(sources.length()-i-1)) {
				_recents_settings.endGroup();
				_recents_settings.remove(sources.at(i));
			}
			else {
				_recents_settings.endGroup();
			}
		}
		_recents_settings.endGroup();
	}
	else {
		QString item_key = _recents_items_keys.value(category);
		QStringList in_list = _recents_settings.value(item_key).toStringList();
		QStringList out_list;
		int index = 0;
		for (const QString& s: in_list) {
			if (!indexes.contains(index)) {
				out_list << s;
			}
			index++;
		}
		_recents_settings.setValue(item_key, indexes.isEmpty() ? QStringList() : out_list);
	}
	_recents_settings.sync();
}

const PVCore::PVRecentItemsManager::variant_list_t PVCore::PVRecentItemsManager::get_list(Category category)
{
	PVCore::PVRecentItemsManager::variant_list_t result;

	switch (category)
	{
		case Category::SUPPORTED_FORMATS:
		{
			result = supported_format_list();
			break;
		}
		case Category::SOURCES:
		{
			result = sources_description_list();
			break;
		}
		case Category::PROJECTS:
		case Category::USED_FORMATS:
		case Category::EDITED_FORMATS:
		{
			result = items_list(category);
			break;
		}
		default:
		{
			assert(false); // Unknown category
			break;
		}
	}

	return result;
}

const PVCore::PVRecentItemsManager::variant_list_t PVCore::PVRecentItemsManager::items_list(Category category, bool update_removed_files /* = false */) const
{
	variant_list_t variant_list;

	QStringList string_list = _recents_settings.value(_recents_items_keys[category]).toStringList();
	QStringList out_string_list;
	for (QString s: string_list) {
		variant_list << QVariant(s);
		if (QFile::exists(s)) {
			out_string_list << s;
		}
	}
	if (update_removed_files) {
		_recents_settings.setValue(_recents_items_keys[category], out_string_list);
		_recents_settings.sync();
	}

	return variant_list;
}

const PVCore::PVRecentItemsManager::variant_list_t PVCore::PVRecentItemsManager::sources_description_list() const
{
	variant_list_t variant_list;

	_recents_settings.beginGroup(_recents_items_keys[SOURCES]);

	QStringList sources = _recents_settings.childGroups();
	for (uint64_t i = sources.length(); i --> 0; ) {
		_recents_settings.beginGroup(sources.at(i));

		PVRush::PVSourceDescription src_desc = deserialize_source_description();
		if(src_desc.is_valid()) {
			QVariant var;
			var.setValue(src_desc);
			variant_list << var;
			_recents_settings.endGroup();
		}
		else {
			_recents_settings.endGroup();
			_recents_settings.remove(sources.at(i));
		}
	}

	_recents_settings.endGroup();
	_recents_settings.sync();

	return variant_list;
}

void PVCore::PVRecentItemsManager::clear_missing_files()
{
	sources_description_list();
	items_list(Category::PROJECTS, true);
	items_list(Category::USED_FORMATS, true);
	items_list(Category::EDITED_FORMATS, true);
}

const PVCore::PVRecentItemsManager::variant_list_t PVCore::PVRecentItemsManager::supported_format_list() const
{
	variant_list_t variant_list;

	LIB_CLASS(PVRush::PVInputType) &input_types = LIB_CLASS(PVRush::PVInputType)::get();
	LIB_CLASS(PVRush::PVInputType)::list_classes const& lf = input_types.get_list();

	LIB_CLASS(PVRush::PVInputType)::list_classes::const_iterator it;

	for (it = lf.begin(); it != lf.end(); it++) {
		PVRush::PVInputType_p in = it.value();

		PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in);
		PVRush::hash_format_creator format_creator = PVRush::PVSourceCreatorFactory::get_supported_formats(lcr);

		PVRush::hash_format_creator::const_iterator itfc;
		for (itfc = format_creator.begin(); itfc != format_creator.end(); itfc++) {
			QVariant var;
			var.setValue(itfc.value().first);
			variant_list << var;
		}
	}

	return variant_list;
}

uint64_t PVCore::PVRecentItemsManager::get_source_timestamp_to_replace(const PVRush::PVSourceDescription& source_description)
{
	QStringList sources = _recents_settings.childGroups();

	uint64_t older_timestamp = QDateTime::currentDateTime().toMSecsSinceEpoch();
	for (QString source_timestamp : sources) {

		older_timestamp = std::min(older_timestamp, (uint64_t) source_timestamp.toULong());

		_recents_settings.beginGroup(source_timestamp);

		PVRush::PVSourceDescription src_desc = deserialize_source_description();

		_recents_settings.endGroup();

		if (source_description == src_desc) {
			return source_timestamp.toULong();
		}
	}

	if (sources.size() < _max_recent_items) return 0;

	return older_timestamp;
}

PVRush::PVSourceDescription PVCore::PVRecentItemsManager::deserialize_source_description() const
{
	// source creator
	QString source_creator_name = _recents_settings.value(ITEM_SUBKEY_SOURCE_CREATOR_NAME).toString();
	PVRush::PVSourceCreator_p src_creator_p = LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name(source_creator_name);
	PVRush::PVInputType_p input_type_p = src_creator_p->supported_type_lib();

	// inputs
	PVRush::PVInputType::list_inputs inputs;
	uint64_t nb_inputs = _recents_settings.beginReadArray(ITEM_SUBKEY_INPUTS);
	for (uint64_t j = 0; j < nb_inputs; ++j) {
		_recents_settings.setArrayIndex(j);
		inputs << input_type_p->load_input_from_qsettings(_recents_settings);
	}
	_recents_settings.endArray();

	// format
	QString format_name = _recents_settings.value(ITEM_SUBKEY_FORMAT_NAME).toString();
	QString format_path = _recents_settings.value(ITEM_SUBKEY_FORMAT_PATH).toString();

	PVRush::PVFormat format(format_name, format_path);

	// PVSourceDescription
	PVRush::PVSourceDescription src_desc(inputs, src_creator_p, format);

	return src_desc;
}

