//! \file PVClassLibrary.h
//! $Id: PVClassLibrary.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVCLASSLIBRARY_H
#define PVCORE_PVCLASSLIBRARY_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVTag.h>

#include <QHash>
#include <QString>

#include <cassert>
#include <algorithm>
#include <typeinfo>

namespace PVCore {

/*! \brief Template class library used to register relevant classes, with associated tags if necessary. used by the plugin system.
 *  \tparam RegAs Kind of class library. Must be a base class of T (see \ref register_class). This is generally the interface of a plugin type.
 *
 * This class is used by the plugin system to keep track of the different registered plugin. Each plugin implements RegAs (or one of its children),
 * and is registered thanks to the REGISTER_CLASS or REGISTER_CLASS_AS macros.
 *
 * The REGISTER_CLASS(desc, T) macro can be used when T defines the type T::RegAs. Indeed, these two codes are equivalent:
 *
 * \code
 * REGISTER_CLASS("plugin-name", MyMappingPlugin);
 * // is the same as
 * PVCore::PVClassLibrary<MyMappingPlugin::RegAs>::get().register_class("plugin-name", MyMappingPlugin());
 * \endcode
 *
 * Note that you need a valid default constructor for this macro to work.
 * You can also use the REGISTER_CLASS_WITH_ARGS macros that helps you tune your registration. Both codes are equivalent:
 *
 * \code
 * REGISTER_CLASS_WITH_ARGS("plugin-name", MyMappingPlugin, arg1, arg2, ...);
 * // is the same as
 * PVCore::PVClassLibrary<MyMappingPlugin::RegAs>::get().register_class("plugin-name", MyMappingPlugin(arg1, arg2, ...));
 * \endcode
 *
 * This technique is used so that, for instance, the same plugin can be registered with different default parameters (in terms of PVArgumentList)
 * and eventually using different constructors.
 *
 * For instance, if you have a plugin implementation like this:
 *
 * \code
 * class MyMappingPlugin: public Picviz::PVMappingFilter
 * {
 * public:
 *     MyMappingPlugin(PVCore::PVArgumentList args = MyMappingPlugin::default_args());
 *     MyMappingPlugin(int param_construct, PVCore::PVArgumentList args = MyMappingPlugin::default_args());
 * [...]
 *     CLASS_REGISTRABLE(MyMappingPlugin)
 * };
 * \endcode
 *
 * then, it can be registered by two ways:
 *
 * \code
 * REGISTER_CLASS("my-plugin-1", MyMappingPlugin);
 * REGISTER_CLASS_WITH_ARGS("my-plugin-2", MyMappingPlugin, 4);
 *
 * In the end, it looks like two plugins exists, but they are using the same underlying class.
 *
 * \note
 * AG: WARNING: there is *no* LibKernelDecl and this is *wanted* !
 *              check the wiki for more informations
 */
template<class RegAs>
class PVClassLibrary {
public:
	// PF is a shared pointer to a registered class's base class
	typedef typename RegAs::p_type PF;
	typedef QHash<QString,PF> list_classes;
	typedef PVClassLibrary<RegAs> C;
	typedef PVTag<RegAs> tag;
	typedef QList<tag> list_tags;

private:
	PVClassLibrary()
	{
		_last_registered_id = 0;
	}

public:
	static C& get()
	{
		static C obj;
		return obj;
	}

public:
	template<class T>
	void register_class(QString const& name, T const& f)
	{
		PF pf = f.template clone<RegAs>();
		pf->__registered_class_name = name;
		pf->__registered_class_id = _last_registered_id;
		_last_registered_id++;
		_classes.insert(name, pf);
	}

	template<class T>
	void declare_tag(QString const& name, QString const& desc)
	{
		// Looks for a registered version of 'T', and take it if it exists
		typename list_classes::iterator it_c;
		PF pf;
		for (it_c = _classes.begin(); it_c != _classes.end(); it_c++) {
			if (dynamic_cast<T*>(it_c.value().get()) != NULL) {
				pf = it_c.value();
				break;
			}
		}
		// If this assert fails, it means that 'T' hasn't been previously registered as 'RegAs' (see REGISTER_CLASS)
		assert(pf);
		typename list_tags::iterator it = std::find(_tags.begin(), _tags.end(), tag(name, ""));
		if (it == _tags.end()) {
			tag new_tag(name, desc);
			new_tag.add_class(pf);
			_tags.push_back(new_tag);
		}
		else {
			tag& cur_tag = *it;
			cur_tag.add_class(pf);
		}
	}

	list_classes const& get_list() const { return _classes; }

	list_tags const& get_tags() const { return _tags; }

	template <class T>
	list_tags get_tags_for_class(T const& f) const
	{
		list_tags ret;
		typename list_tags::const_iterator it;
		for (it = _tags.begin(); it != _tags.end(); it++) {
			tag const& t = *it;
			typename tag::list_classes const& lc = t.associated_classes();
			typename tag::list_classes::const_iterator it_c;
			bool found = false;
			for (it_c = lc.begin(); it_c != lc.end(); it_c++) {
				if (typeid(*(*it_c)) == typeid(f)) {
					found = true;
					break;
				}
			}
			if (!found) {
				continue;
			}
			ret.push_back(t);
		}
		return ret;
	}

	tag const& get_tag(QString name)
	{
		typename list_tags::const_iterator it = std::find(_tags.begin(), _tags.end(), tag(name, ""));
		if (it == _tags.end()) {
			throw PVTagUndefinedException(name);
		}

		return *it;
	}

	// A shared pointer is returned, which means that parameters can be saved accross this
	// saved pointer. If this is not wanted, a clone can be made thanks to the clone() method
	PF get_class_by_name(QString const& name) const
	{
		if (!_classes.contains(name))
			return PF();
		return _classes[name];
	}

private:
	list_classes _classes;
	list_tags _tags;
	int _last_registered_id;
};

class LibKernelDecl PVClassLibraryLibLoader {
public:
	static bool load_class(QString const& path);
	static int load_class_from_dir(QString const& pluginsdir, QString const& prefix);

	/**
	 * Load plugins from several directories
	 *
	 * @param pluginsdirs list of directories separated with the semicolon char ';'
	 * @param prefix plugin type prefix, such as "plotting_filter" for the plotting filter plugin type
	 */
	static int load_class_from_dirs(QString const& pluginsdirs, QString const& prefix);
	static QStringList split_plugin_dirs(QString const& dirs);
};

#define REGISTER_CLASS_AS(name, T, RegAs) \
	PVCore::PVClassLibrary<RegAs>::get().register_class<T>(name, T());
#define REGISTER_CLASS(name, T) REGISTER_CLASS_AS(name, T, T::RegAs)
	
#define REGISTER_CLASS_AS_WITH_ARGS(name, T, RegAs, ...) \
	PVCore::PVClassLibrary<RegAs>::get().register_class<T>(name, T(__VA_ARGS__));
#define REGISTER_CLASS_WITH_ARGS(name, T, ...) REGISTER_CLASS_AS_WITH_ARGS(name, T, T::RegAs, __VA_ARGS__)

#define DECLARE_TAG_AS(name, desc, T, RegAs) \
	PVCore::PVClassLibrary<RegAs>::get().declare_tag<T>(name, desc);
#define DECLARE_TAG(name, desc, T) DECLARE_TAG_AS(name, desc, T, T::RegAs)

#define LIB_CLASS(T) \
	PVCore::PVClassLibrary<T::RegAs>
}

#endif
