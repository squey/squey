/**
 * \file PVPluginPresets.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVPLUGIN_PRESETS_H
#define PVCORE_PVPLUGIN_PRESETS_H

#include <pvkernel/core/PVFunctionArgs.h>

namespace PVCore {

namespace __impl {

class PVPluginPresets
{
public:
	PVPluginPresets(PVFunctionArgsBase* fargs, QString const& registered_name, QString const& path);

	QStringList list_presets() const;
	void del_preset(const QString& name) const;

	void add_preset(const QString& name) const;
	void load_preset(const QString& name);
	void modify_preset(const QString& name) const;
	void rename_preset(QString const& old_name, QString const& new_name) const;

	PVArgumentList get_args_for_preset() const;
	bool can_have_presets() const;

private:
	PVFunctionArgsBase* _fargs;
	const QString&      _registered_name;
	const QString&      _path;
	QString     	    _abs_reg_name;
};

}

template <class T>
class PVPluginPresets: public __impl::PVPluginPresets
{
public:
	PVPluginPresets(T& o, QString const& path):
		__impl::PVPluginPresets(dynamic_cast<PVFunctionArgsBase*>(&o), o.registered_name(), path)
	{ }
};

}

#endif // PVCORE_PVPLUGIN_PRESETS_H
