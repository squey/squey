/*
 * \file PVPluginPresets.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVPluginPresets.h>

#include <QStringList>
#include <QFileInfo>
#include <QDir>

#include <memory>

#define PRESETS_FILENAME "presets.ini"

typedef std::shared_ptr<QSettings> QSettings_p;

/* using a static shared pointer instead of a static pointer on the QSettings
 * insures its destructor is called on program termination (because it is in
 * the data segment). It garantees the INI file is properly closed.
 */
static QSettings_p g_presets_settings;

PVCore::__impl::PVPluginPresets::PVPluginPresets(PVCore::PVFunctionArgsBase* fargs, QString const& registered_name, QString const& path)
 : _fargs(fargs)
 , _registered_name(registered_name)
 , _path(path)
 , _abs_reg_name(path + "/" + registered_name)
{
	if (g_presets_settings.get() == nullptr) {
		// ("presets.ini", QSettings::IniFormat);
		// merde !

		QFileInfo fi(QDir::homePath() + QDir::separator() + PICVIZ_INSPECTOR_CONFDIR + QDir::separator() + PRESETS_FILENAME);

		if (fi.exists() == false) {
			fi.dir().mkpath(fi.path());

			QFileInfo sys_fi(PRESETS_FILENAME);

			if (sys_fi.exists()) {
				QFile::copy(sys_fi.filePath(), fi.filePath());
			}
		}

		g_presets_settings = QSettings_p(new QSettings(fi.filePath(), QSettings::IniFormat));
	}
}

QStringList PVCore::__impl::PVPluginPresets::list_presets() const
{
	g_presets_settings->beginGroup(_abs_reg_name);
	QStringList presets = g_presets_settings->childGroups();
	g_presets_settings->endGroup();

	return presets;
}

void PVCore::__impl::PVPluginPresets::del_preset(QString const& name) const
{
	g_presets_settings->remove(_abs_reg_name + "/" + name);
}

void PVCore::__impl::PVPluginPresets::add_preset(QString const& name) const
{
	PVArgumentList_to_QSettings(get_args_for_preset(), *g_presets_settings, _abs_reg_name + "/" + name);
}

void PVCore::__impl::PVPluginPresets::modify_preset(QString const& name) const
{
	del_preset(name);
	add_preset(name);
}

void PVCore::__impl::PVPluginPresets::rename_preset(QString const& old_name, QString const& new_name) const
{
	// too bad that QSettings doesn't have a rename method: must remove and re-add preset...
	PVArgumentList args = PVCore::QSettings_to_PVArgumentList(*g_presets_settings, _fargs->get_default_args(), _abs_reg_name + "/" + old_name);
	del_preset(old_name);
	PVArgumentList_to_QSettings(get_args_for_preset(), *g_presets_settings, _abs_reg_name + "/" + new_name);
}


void PVCore::__impl::PVPluginPresets::load_preset(QString const& name)
{
	PVArgumentList args = PVCore::QSettings_to_PVArgumentList(*g_presets_settings, _fargs->get_default_args(), _abs_reg_name + "/" + name);

	_fargs->set_args_from_preset(args);
}

PVCore::PVArgumentList PVCore::__impl::PVPluginPresets::get_args_for_preset() const
{
	return _fargs->get_args_for_preset();
}

bool PVCore::__impl::PVPluginPresets::can_have_presets() const
{
	return _fargs->get_args_keys_for_preset().count() > 0;
}
