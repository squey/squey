/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVFormat.h>           // for PVFormat
#include <pvkernel/rush/PVInputDescription.h> // for PVInputDescription
#include <pvkernel/rush/PVInputType.h>        // for PVInputType, etc
#include <pvkernel/rush/PVSourceCreator.h>    // for PVSourceCreator, etc
#include <pvkernel/rush/PVSourceCreatorFactory.h>
#include <pvkernel/rush/PVSourceDescription.h> // for PVSourceDescription

#include <pvkernel/core/PVRecentItemsManager.h>
#include "pvkernel/core/PVClassLibrary.h" // for LIB_CLASS, etc
#include "pvkernel/core/PVOrderedMap.h"
#include "pvkernel/core/PVRegistrableClass.h"

#include <pvbase/general.h>

#include <cassert>   // for assert
#include <cstdint>   // for uint64_t
#include <algorithm> // for min
#include <memory>    // for __shared_ptr, shared_ptr
#include <utility>   // for pair

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QList>
#include <QSettings>
#include <QString>
#include <QStringList>

#define RECENTS_FILENAME "recents.ini"

constexpr const char* ITEM_SUBKEY_SOURCE_CREATOR_NAME = "__sc_name";
constexpr const char* ITEM_SUBKEY_FORMAT_NAME = "__format_name";
constexpr const char* ITEM_SUBKEY_FORMAT_PATH = "__format_path";
constexpr const char* ITEM_SUBKEY_INPUTS = "inputs";

void PVCore::PVRecentItemsManager::add_source(PVRush::PVSourceCreator_p source_creator_p,
                                              const PVRush::PVInputType::list_inputs& inputs,
                                              const PVRush::PVFormat& format)
{
	_recents_settings.beginGroup(_recents_items_keys[SOURCES]);

	// Look for the timestamp to replace
	uint64_t source_timestamp = get_source_timestamp_to_replace(
	    PVRush::PVSourceDescription(inputs, source_creator_p, format));
	if (source_timestamp) {
		_recents_settings.remove(QString::number(source_timestamp));
	}

	// Set source description information in it.
	_recents_settings.beginGroup(QString::number(QDateTime::currentDateTime().toMSecsSinceEpoch()));

	_recents_settings.setValue(ITEM_SUBKEY_SOURCE_CREATOR_NAME,
	                           source_creator_p->registered_name());
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
	_add_item[Category::SOURCES].emit();
}

void PVCore::PVRecentItemsManager::clear(Category category, QList<int> indexes)
{
	if (category == Category::SOURCES) {
		_recents_settings.beginGroup(_recents_items_keys[Category::SOURCES]);
		QStringList sources = _recents_settings.childGroups();
		for (int i = sources.length(); i-- > 0;) {
			_recents_settings.beginGroup(sources.at(i));
			if (indexes.isEmpty() || indexes.contains(sources.length() - i - 1)) {
				_recents_settings.endGroup();
				_recents_settings.remove(sources.at(i));
			} else {
				_recents_settings.endGroup();
			}
		}
		_recents_settings.endGroup();
	} else {
		QString item_key = _recents_items_keys.value(category);
		QStringList in_list = _recents_settings.value(item_key).toStringList();
		QStringList out_list;
		int index = 0;
		for (const QString& s : in_list) {
			if (!indexes.contains(index)) {
				out_list << s;
			}
			index++;
		}
		_recents_settings.setValue(item_key, indexes.isEmpty() ? QStringList() : out_list);
	}
	_recents_settings.sync();
}

template <PVCore::Category category>
typename PVCore::list_type<category>::type PVCore::PVRecentItemsManager::items_list() const
{
	QStringList v = _recents_settings.value(_recents_items_keys[category]).toStringList();
	std::vector<std::string> res(v.size());
	std::transform(v.begin(), v.end(), res.begin(), std::mem_fun_ref(&QString::toStdString));
	return res;
}

void PVCore::PVRecentItemsManager::remove_missing_files(Category category)
{
	QStringList string_list = _recents_settings.value(_recents_items_keys[category]).toStringList();
	QStringList out_string_list;
	for (QString s : string_list) {
		if (QFile::exists(s)) {
			out_string_list << s;
		}
	}
	_recents_settings.setValue(_recents_items_keys[category], out_string_list);
	_recents_settings.sync();
}

void PVCore::PVRecentItemsManager::remove_invalid_source()
{
	_recents_settings.beginGroup(_recents_items_keys[SOURCES]);

	QStringList sources = _recents_settings.childGroups();
	for (QString source : sources) {
		_recents_settings.beginGroup(source);

		try {
			PVRush::PVSourceDescription src_desc(deserialize_source_description());
			_recents_settings.endGroup();
		} catch (PVRush::BadInputDescription const& e) {
			_recents_settings.endGroup();
			_recents_settings.remove(source);
		} catch (PVCore::InvalidPlugin const& e) {
			_recents_settings.endGroup();
			_recents_settings.remove(source);
		} catch (PVRush::PVInvalidFile const& e) {
			_recents_settings.endGroup();
			_recents_settings.remove(source);
		}
	}

	_recents_settings.endGroup();
	_recents_settings.sync();
}

std::vector<PVCore::PVSerializedSource> PVCore::PVRecentItemsManager::sources_description_list()
{
	std::vector<PVCore::PVSerializedSource> res;

	_recents_settings.beginGroup(_recents_items_keys[SOURCES]);

	QStringList sources = _recents_settings.childGroups();
	for (int i = sources.length(); i-- > 0;) {
		QString source = sources.at(i);
		_recents_settings.beginGroup(source);

		try {
			res.emplace_back(deserialize_source_description());
		} catch (PVRush::BadInputDescription const& e) {
			// Input description is invalid
		} catch (PVCore::InvalidPlugin const& e) {
			// If the plugin is incorrect, skip this file
		} catch (PVRush::PVInvalidFile const& e) {
			// If a format can't be found, skip this source.
		}

		_recents_settings.endGroup();
	}

	_recents_settings.endGroup();

	return res;
}

void PVCore::PVRecentItemsManager::clear_missing_files()
{
	remove_invalid_source();
	remove_missing_files(Category::PROJECTS);
	remove_missing_files(Category::USED_FORMATS);
	remove_missing_files(Category::EDITED_FORMATS);
}

std::vector<std::string> PVCore::PVRecentItemsManager::supported_format_list() const
{
	std::vector<std::string> res;

	LIB_CLASS(PVRush::PVInputType)& input_types = LIB_CLASS(PVRush::PVInputType)::get();
	LIB_CLASS(PVRush::PVInputType)::list_classes const& lf = input_types.get_list();

	for (const auto& it : lf) {
		PVRush::PVInputType_p in = it.value();

		PVRush::list_creators lcr = PVRush::PVSourceCreatorFactory::get_by_input_type(in);
		PVRush::hash_format_creator format_creator =
		    PVRush::PVSourceCreatorFactory::get_supported_formats(lcr);

		for (auto itfc = format_creator.begin(); itfc != format_creator.end(); itfc++) {
			res.emplace_back(itfc.value().first.get_full_path().toStdString());
		}
	}

	return res;
}

uint64_t PVCore::PVRecentItemsManager::get_source_timestamp_to_replace(
    const PVRush::PVSourceDescription& source_description)
{
	QStringList sources = _recents_settings.childGroups();

	uint64_t older_timestamp = QDateTime::currentDateTime().toMSecsSinceEpoch();
	for (QString source_timestamp : sources) {

		older_timestamp = std::min(older_timestamp, (uint64_t)source_timestamp.toULong());

		_recents_settings.beginGroup(source_timestamp);

		try {
			PVRush::PVSourceDescription src_desc(deserialize_source_description());
			if (source_description == src_desc) {
				_recents_settings.endGroup();
				return source_timestamp.toULong();
			}
		} catch (PVRush::BadInputDescription const& e) {
			// If any input is invalid, it can't be the searched as a source to replace.
		} catch (PVCore::InvalidPlugin const& e) {
			// If the plugin is invalid, it can't be the searched as a source to replace.
		} catch (PVRush::PVInvalidFile const& e) {
			// If the format is invalid, it can't be the searched as a source to replace.
		}

		_recents_settings.endGroup();
	}

	if (sources.size() < _max_recent_items) {
		return 0;
	}

	return older_timestamp;
}

PVCore::PVSerializedSource PVCore::PVRecentItemsManager::deserialize_source_description()
{
	// source creator
	PVCore::PVSerializedSource seri_src{
	    {},
	    _recents_settings.value(ITEM_SUBKEY_SOURCE_CREATOR_NAME).toString().toStdString(),
	    _recents_settings.value(ITEM_SUBKEY_FORMAT_NAME).toString().toStdString(),
	    _recents_settings.value(ITEM_SUBKEY_FORMAT_PATH).toString().toStdString()};

	// get inputs data
	PVRush::PVSourceCreator_p src_creator_p =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name(
	        QString::fromStdString(seri_src.sc_name));
	PVRush::PVInputType_p input_type_p = src_creator_p->supported_type_lib();
	uint64_t nb_inputs = _recents_settings.beginReadArray(ITEM_SUBKEY_INPUTS);
	try {
		for (uint64_t j = 0; j < nb_inputs; ++j) {
			_recents_settings.setArrayIndex(j);
			seri_src.input_desc.emplace_back(
			    input_type_p->load_input_descr_from_qsettings(_recents_settings));
		}
		_recents_settings.endArray();
	} catch (...) {
		_recents_settings.endArray();
		throw;
	}

	return seri_src;
}

static QString get_recent_items_file()
{
	QFileInfo fi(QDir::homePath() + QDir::separator() + INENDI_INSPECTOR_CONFDIR +
	             QDir::separator() + RECENTS_FILENAME);

	if (fi.exists() == false) {
		fi.dir().mkpath(fi.path());

		QFileInfo sys_fi(RECENTS_FILENAME);

		if (sys_fi.exists()) {
			QFile::copy(sys_fi.filePath(), fi.filePath());
		}
	}

	return fi.filePath();
}

PVCore::PVRecentItemsManager::PVRecentItemsManager()
    : _recents_settings(get_recent_items_file(), QSettings::IniFormat)
{
	clear_missing_files();
}

std::tuple<QString, QStringList>
PVCore::PVRecentItemsManager::get_string_from_entry(QString const& string) const
{
	return std::make_tuple(string, QStringList() << string);
}

std::tuple<QString, QStringList> PVCore::PVRecentItemsManager::get_string_from_entry(
    PVCore::PVSerializedSource const& src_desc) const
{
	QString long_string;
	QStringList filenames;

	if (src_desc.input_desc.size() == 1) {
		PVRush::PVSourceDescription input(src_desc);
		QString source_path = input.get_inputs()[0]->human_name();
		long_string = source_path + " [" + QString::fromStdString(src_desc.format_name) + "]";
		filenames << source_path;
	} else {
		PVRush::PVSourceDescription desc(src_desc);
		for (auto const& input : desc.get_inputs()) {
			filenames << input->human_name();
		}
		long_string =
		    "[" + QString::fromStdString(src_desc.format_name) + "]\n" + filenames.join("\n");
	}

	return std::make_tuple(long_string, filenames);
}

std::tuple<QString, QStringList>
PVCore::PVRecentItemsManager::get_string_from_entry(PVRush::PVFormat const& format) const
{
	QString long_string =
	    QString("%1 (%2)").arg(format.get_format_name()).arg(format.get_full_path());
	QStringList filenames;
	filenames << format.get_full_path();

	return std::make_tuple(long_string, filenames);
}

namespace PVCore
{

template <>
typename list_type<Category::PROJECTS>::type PVRecentItemsManager::get_list<Category::PROJECTS>()
{
	return items_list<Category::PROJECTS>();
}

template <>
typename list_type<Category::USED_FORMATS>::type
PVRecentItemsManager::get_list<Category::USED_FORMATS>()
{
	return items_list<Category::USED_FORMATS>();
}

template <>
typename list_type<Category::EDITED_FORMATS>::type
PVRecentItemsManager::get_list<Category::EDITED_FORMATS>()
{
	return items_list<Category::EDITED_FORMATS>();
}

template <>
typename list_type<Category::SUPPORTED_FORMATS>::type
PVRecentItemsManager::get_list<Category::SUPPORTED_FORMATS>()
{
	return supported_format_list();
}

template <>
typename list_type<Category::SOURCES>::type PVRecentItemsManager::get_list<Category::SOURCES>()
{
	return sources_description_list();
}

} // namespace PVCore
