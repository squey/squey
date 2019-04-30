/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDISPLAYS_PVDISPLAYSIMPL_H
#define PVDISPLAYS_PVDISPLAYSIMPL_H

#include <pvdisplays/PVDisplayIf.h>

#include <QMetaType>

namespace PVDisplays
{

class PVDisplaysContainer;

class PVDisplaysImpl : public QObject
{
  private:
	PVDisplaysImpl()
	{
		// Load all plugins
		load_plugins();
	}

  public:
	static PVDisplaysImpl& get();

  public:
	template <typename If, typename F>
	static void visit_displays_by_if(F const& f, int flags = 0)
	{
		// Interface of 'F' must be void f(If& obj), or with a base of If;
		typename PVCore::PVClassLibrary<If>::list_classes const& lc =
		    PVCore::PVClassLibrary<If>::get().get_list();
		// `lc' is of type QHash<QString,shared_pointer<If>>
		for (auto it = lc.begin(); it != lc.end(); it++) {
			If& obj = *(it->value());
			if (obj.match_flags(flags)) {
				f(obj);
			}
		}
	}

	template <typename If, typename... P>
	static QWidget* get_widget(If& interface, P&&... args)
	{
		if (interface.match_flags(PVDisplayIf::UniquePerParameters)) {
			return interface.get_unique_widget(std::forward<P>(args)...);
		}

		return interface.create_widget(std::forward<P>(args)...);
	}

	static void add_displays_view_axis_menu(QMenu& menu,
	                                        PVDisplaysContainer* container,
	                                        Inendi::PVView* view,
	                                        PVCombCol axis_comb);

  private:
	void load_plugins();

  private:
	static PVDisplaysImpl* _instance;
};

} // namespace PVDisplays

#endif
