/**
 * \file PVRecentItemsManager.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVRECENTITEMSMANAGER_H_
#define PVRECENTITEMSMANAGER_H_

#include <QObject>
#include <QSettings>
#include <QStringList>
#include <QDateTime>
#include <QList>
#include <QVariant>

#include <pvkernel/core/PVSharedPointer.h>

#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVSourceCreatorFactory.h>
#include <pvkernel/rush/PVSourceDescription.h>
#include <pvkernel/rush/PVFormat.h>

Q_DECLARE_METATYPE(PVRush::PVSourceDescription)
Q_DECLARE_METATYPE(PVRush::PVFormat)

namespace PVCore
{

class PVRecentItemsManager
{
public:
	typedef PVCore::PVSharedPtr<PVRecentItemsManager> PVRecentItemsManager_p;
	typedef QList<QVariant> variant_list_t;

public:
	enum Category {
		FIRST = 0,

		PROJECTS = FIRST,
		SOURCES,
		USED_FORMATS,
		EDITED_FORMATS,
		SUPPORTED_FORMATS,

		LAST
	};

	static PVRecentItemsManager_p& get()
	{
		if (_recent_items_manager_p.get() == nullptr) {
			_recent_items_manager_p = PVRecentItemsManager_p(new PVRecentItemsManager());
		}
		return _recent_items_manager_p;
	}

	const QString get_key(Category category)
	{
		return _recents_items_keys[category];
	}

	void add(const QString& item_path, Category category)
	{
		QString recent_items_key = _recents_items_keys[category];

		QStringList files = _recents_settings.value(recent_items_key).toStringList();
		files.removeAll(item_path);
		files.prepend(item_path);
		for (; files.size() > _max_recent_items; files.removeLast()) {}
		_recents_settings.setValue(recent_items_key, files);
	}

	void add_source(PVRush::PVSourceCreator_p source_creator_p, const PVRush::PVInputType::list_inputs& inputs, PVRush::PVFormat& format)
	{
		int index = _recents_settings.beginReadArray(_recents_items_keys[SOURCES]);
		_recents_settings.endArray();

		/*if (index == _max_recent_items) {
			uint64_t older_timestamp = QDateTime::currentDateTime().toTime_t();
			uint64_t older_timstamp_index = 0;
			for (int i = 1; i <= _max_recent_items; i++) {
				uint64_t timestamp = _recents_settings.value(QString("%1/%2/__timestamp").arg(_recents_items_keys[SOURCES]).arg(i)).toInt();
				if (timestamp < older_timestamp) {
					older_timestamp = timestamp;
					older_timstamp_index = i;
				}
			}
			_recents_settings.remove(QString("%1/%2").arg(_recents_items_keys[SOURCES]).arg(older_timstamp_index));
			index--;
		}*/

		_recents_settings.beginWriteArray(_recents_items_keys[SOURCES]);
		_recents_settings.setArrayIndex(index);
		_recents_settings.setValue("__sc_name", source_creator_p->registered_name());
		_recents_settings.setValue("__format", format.get_full_path());
		_recents_settings.setValue("__timestamp", QDateTime::currentDateTime().toTime_t());
			_recents_settings.beginWriteArray("inputs");
			int inputs_index = 0;
			for (auto input : inputs) {
				_recents_settings.setArrayIndex(inputs_index++);
				input->save_to_qsettings(_recents_settings);
			}
			_recents_settings.endArray();
		_recents_settings.endArray();
	}

	const variant_list_t get_list(Category category)
	{
		switch (category)
		{
			case Category::SUPPORTED_FORMATS:
			{
				return supported_format_list();
			}
			case Category::SOURCES:
			{
				return sources_description_list();
			}
			case Category::PROJECTS:
			case Category::USED_FORMATS:
			case Category::EDITED_FORMATS:
			{
				return items_list(category);
			}
			default:
			{
				assert(false); // Unknown category
				break;
			}
		}
	}

private:
	const variant_list_t items_list(Category category) const
	{
		variant_list_t variant_list;

		QStringList string_list = _recents_settings.value(_recents_items_keys[category]).toStringList();
		for (QString s: string_list) {
			variant_list << QVariant(s);
		}

		return variant_list;
	}

	const variant_list_t sources_description_list() const
	{
		variant_list_t variant_list;

		PVRush::PVInputType::list_inputs inputs;

		uint64_t nb_sources = _recents_settings.beginReadArray(_recents_items_keys[SOURCES]);
		for (uint64_t i = 0; i < nb_sources; ++i) {
			_recents_settings.setArrayIndex(i);

			// source creator
			QString source_creator_name = _recents_settings.value("__sc_name").toString();
			std::cout << "source_creator_name=" << source_creator_name.toStdString() << std::endl;
			PVRush::PVSourceCreator_p src_creator_p = LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name(source_creator_name);
			PVRush::PVInputType_p input_type_p = src_creator_p->supported_type_lib();

			// inputs
			uint64_t nb_inputs = _recents_settings.beginReadArray("inputs");
			for (uint64_t j = 0; j < nb_inputs; ++j) {
				_recents_settings.setArrayIndex(j);
				inputs << input_type_p->load_input_from_qsettings(_recents_settings);
			}
			_recents_settings.endArray();

			// format
			QString format_path = _recents_settings.value("__format").toString();
			PVRush::PVFormat format;
			format.populate_from_xml(format_path);

			// PVSourceDescription
			PVRush::PVSourceDescription src_desc(inputs, src_creator_p, format);
			QVariant var;
			var.setValue(src_desc);
			variant_list << var;
		}
		_recents_settings.endArray();

		return variant_list;
	}

	const variant_list_t supported_format_list() const
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

private:
	PVRecentItemsManager() : _recents_settings("recents.ini", QSettings::IniFormat) {}
	PVRecentItemsManager(const PVRecentItemsManager&);
	PVRecentItemsManager &operator=(const PVRecentItemsManager&);

private:
	static PVRecentItemsManager_p _recent_items_manager_p;

	mutable QSettings _recents_settings;
	const int64_t _max_recent_items = 2;
	const QStringList _recents_items_keys = { "recent_projects", "recent_sources", "recent_used_formats", "recent_edited_formats", "supported_formats" };
};

}

#endif /* PVRECENTITEMSMANAGER_H_ */
