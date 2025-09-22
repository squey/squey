/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVCORE_PVCLASSLIBRARY_H
#define PVCORE_PVCLASSLIBRARY_H

#include <pvkernel/core/PVOrderedMap.h> // for PVOrderedMap
#include <pvkernel/core/PVSingleton.h>
#include <qcontainerfwd.h>
#include <cassert>   // for assert
#include <stdexcept> // for runtime_error
#include <string>    // for operator+, basic_string
#include <typeinfo>  // for type_info
#include <QString>
#include <QStringList>
#include <QList>

#include <pvlogger.h>

namespace PVCore
{

class InvalidPlugin : public std::runtime_error
{
	using std::runtime_error::runtime_error;
};

/*! \brief Template class library used to register relevant classes.
 *  \tparam RegAs Kind of class library. Must be a base class of T (see \ref register_class). This
 *is generally the interface of a plugin type.
 *
 * This class is used by the plugin system to keep track of the different registered plugin. Each
 *plugin implements RegAs (or one of its children),
 * and is registered thanks to the REGISTER_CLASS or REGISTER_CLASS_AS macros.
 *
 * The REGISTER_CLASS(desc, T) macro can be used when T defines the type T::RegAs. Indeed, these two
 *codes are equivalent:
 *
 * \code
 * REGISTER_CLASS("plugin-name", MyMappingPlugin);
 * // is the same as
 * PVCore::PVClassLibrary<MyMappingPlugin::RegAs>::get().register_class("plugin-name",
 *MyMappingPlugin());
 * \endcode
 *
 * Note that you need a valid default constructor for this macro to work.
 * You can also use the REGISTER_CLASS_WITH_ARGS macros that helps you tune your registration. Both
 *codes are equivalent:
 *
 * \code
 * REGISTER_CLASS_WITH_ARGS("plugin-name", MyMappingPlugin, arg1, arg2, ...);
 * // is the same as
 * PVCore::PVClassLibrary<MyMappingPlugin::RegAs>::get().register_class("plugin-name",
 *MyMappingPlugin(arg1, arg2, ...));
 * \endcode
 *
 * This technique is used so that, for instance, the same plugin can be registered with different
 *default parameters (in terms of PVArgumentList)
 * and eventually using different constructors.
 *
 * For instance, if you have a plugin implementation like this:
 *
 * \code
 * class MyMappingPlugin: public Squey::PVMappingFilter
 * {
 * public:
 *     MyMappingPlugin(PVCore::PVArgumentList args = MyMappingPlugin::default_args());
 *     MyMappingPlugin(int param_construct, PVCore::PVArgumentList args =
 *MyMappingPlugin::default_args());
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
 * AG: WARNING: there is *no* and this is *wanted* !
 *              check the wiki for more informations
 */

#ifdef _WIN32
	#ifdef BUILD_DLL
		#define SQUEY_EXPORT __attribute__((visibility("hidden")))
	#else
		#define SQUEY_EXPORT __attribute__((visibility("hidden")))
	#endif
#else
	#define SQUEY_EXPORT
#endif



template <class RegAs>
class SQUEY_EXPORT PVClassLibrary
{
  public:
	// PF is a shared pointer to a registered class's base class
	typedef typename RegAs::p_type PF;
	typedef PVCore::PVOrderedMap<QString, PF> list_classes;
	typedef PVClassLibrary<RegAs> C;

	public:
	static C& get()
	{
#if _WIN32
		return PVSingleton<C>::get(); // work across dll
#else
		static C obj;
		return obj;
#endif
	}


  public:
	template <class T>
	void register_class(QString const& name, T const& f)
	{
		PF pf = f.template clone<RegAs>();
		pf->__registered_class_name = name;
		pf->__registered_class_id = _last_registered_id;
		_last_registered_id++;
		_classes[name] = pf;
	}

	list_classes const& get_list() const { return _classes; }

	// A shared pointer is returned, which means that parameters can be saved accross this
	// saved pointer. If this is not wanted, a clone can be made thanks to the clone() method
	PF get_class_by_name(QString const& name) const
	{
		if (!_classes.contains(name)) {
			throw InvalidPlugin("Unknown plugins : " + name.toStdString());
		}
		return _classes.at(name);
	}

  private:
	list_classes _classes;
	int _last_registered_id = 0;
};

class PVClassLibraryLibLoader
{
  public:
	static bool load_class(QString const& path);
	static int load_class_from_dir(QString const& pluginsdir, QString const& prefix);

	/**
	 * Load plugins from several directories
	 *
	 * @param pluginsdirs list of directories separated with the semicolon char ';'
	 * @param prefix plugin type prefix, such as "scaling_filter" for the scaling filter plugin
	 *type
	 */
	static int load_class_from_dirs(QString const& pluginsdirs, QString const& prefix);
	static QStringList split_plugin_dirs(QString const& dirs);
};

#define REGISTER_CLASS_AS(name, T, RegAs)                                                          \
	PVCore::PVClassLibrary<RegAs>::get().register_class<T>(name, T());
#define REGISTER_CLASS(name, T) REGISTER_CLASS_AS(name, T, T::RegAs)

#define REGISTER_CLASS_AS_WITH_ARGS(name, T, RegAs, ...)                                           \
	PVCore::PVClassLibrary<RegAs>::get().register_class<T>(name, T(__VA_ARGS__));
#define REGISTER_CLASS_WITH_ARGS(name, T, ...)                                                     \
	REGISTER_CLASS_AS_WITH_ARGS(name, T, T::RegAs, __VA_ARGS__)

#define LIB_CLASS(T) PVCore::PVClassLibrary<T::RegAs>
} // namespace PVCore

#endif
