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

#include <pvbase/general.h>                // for SQUEY_SQUEY_CONFDIR
#include <pvkernel/core/PVPluginPresets.h> // for PVPluginPresets
#include <memory>                          // for __shared_ptr, operator==, etc

#include <pvkernel/core/PVArgument.h>     // for PVArgumentList_to_QSettings, etc
#include <pvkernel/core/PVFunctionArgs.h> // for PVFunctionArgsBase

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSettings>
#include <QString>
#include <QStringList>

constexpr const char* PRESETS_FILENAME = "presets.ini";

using QSettings_p = std::shared_ptr<QSettings>;

/* using a static shared pointer instead of a static pointer on the QSettings
 * insures its destructor is called on program termination (because it is in
 * the data segment). It garantees the INI file is properly closed.
 */
static QSettings_p g_presets_settings;

PVCore::__impl::PVPluginPresets::PVPluginPresets(PVCore::PVFunctionArgsBase* fargs,
                                                 QString const& registered_name,
                                                 QString const& path)
    : _fargs(fargs), _abs_reg_name(path + "/" + registered_name)
{
	if (g_presets_settings == nullptr) {
		QFileInfo fi(QDir::homePath() + QDir::separator() + SQUEY_SQUEY_CONFDIR +
		             QDir::separator() + PRESETS_FILENAME);

		if (!fi.exists()) {
			fi.dir().mkpath(fi.path());

			QFileInfo sys_fi(PRESETS_FILENAME);

			if (sys_fi.exists()) {
				QFile::copy(sys_fi.filePath(), fi.filePath());
			}
		}

		g_presets_settings = std::make_shared<QSettings>(fi.filePath(), QSettings::IniFormat);
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
	PVArgumentList_to_QSettings(get_args_for_preset(), *g_presets_settings,
	                            _abs_reg_name + "/" + name);
}

void PVCore::__impl::PVPluginPresets::modify_preset(QString const& name) const
{
	del_preset(name);
	add_preset(name);
}

void PVCore::__impl::PVPluginPresets::rename_preset(QString const& old_name,
                                                    QString const& new_name) const
{
	// too bad that QSettings doesn't have a rename method: must remove and re-add preset...
	PVArgumentList args = PVCore::QSettings_to_PVArgumentList(
	    *g_presets_settings, _fargs->get_default_args(), _abs_reg_name + "/" + old_name);
	del_preset(old_name);
	PVArgumentList_to_QSettings(get_args_for_preset(), *g_presets_settings,
	                            _abs_reg_name + "/" + new_name);
}

void PVCore::__impl::PVPluginPresets::load_preset(QString const& name)
{
	PVArgumentList args = PVCore::QSettings_to_PVArgumentList(
	    *g_presets_settings, _fargs->get_default_args(), _abs_reg_name + "/" + name);

	_fargs->set_args_from_preset(args);
}

PVCore::PVArgumentList PVCore::__impl::PVPluginPresets::get_args_for_preset() const
{
	return _fargs->get_args_for_preset();
}

bool PVCore::__impl::PVPluginPresets::can_have_presets() const
{
	return _fargs->get_args_keys_for_preset().empty();
}
