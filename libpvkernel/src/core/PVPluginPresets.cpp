#include <pvkernel/core/PVPluginPresets.h>

#include <QStringList>


PVCore::__impl::PVPluginPresets::PVPluginPresets(PVCore::PVFunctionArgsBase* fargs, QString const& registered_name, QString const& path)
 : _fargs(fargs)
 , _registered_name(registered_name)
 , _path(path)
 , _abs_reg_name(path + "/" + registered_name)
{

}

QStringList PVCore::__impl::PVPluginPresets::list_presets()
{
	pvconfig.beginGroup(_abs_reg_name);
	QStringList presets = pvconfig.childGroups();
	pvconfig.endGroup();

	return presets;
}

void PVCore::__impl::PVPluginPresets::del_preset(QString const& name)
{
	pvconfig.remove(_abs_reg_name + "/" + name);
}

void PVCore::__impl::PVPluginPresets::add_preset(QString const& name)
{
	PVArgumentList_to_QSettings(_fargs->get_args_for_preset(), pvconfig, _abs_reg_name + "/" + name);
}

void PVCore::__impl::PVPluginPresets::modify_preset(QString const& name)
{
	del_preset(name);
	add_preset(name);
}

void PVCore::__impl::PVPluginPresets::load_preset(QString const& name)
{
	PVArgumentList args = PVCore::QSettings_to_PVArgumentList(pvconfig, _fargs->get_default_args(), _abs_reg_name + "/" + name);

	_fargs->set_args_from_preset(args);
}

const PVCore::PVArgumentList& PVCore::__impl::PVPluginPresets::get_args_for_preset()
{
	return _fargs->get_args_for_preset();
}
