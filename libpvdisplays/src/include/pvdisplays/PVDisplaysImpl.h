#ifndef PVDISPLAYS_PVDISPLAYSIMPL_H
#define PVDISPLAYS_PVDISPLAYSIMPL_H

#include <pvdisplays/PVDisplayIf.h>

namespace PVDisplays {

class PVDisplaysImpl
{
private:
	PVDisplaysImpl()
	{
		// Load all plugins
		load_plugins();

		// Static initialisation of everyone!
		static_init();
	}

	~PVDisplaysImpl()
	{
		static_release();
	}

public:
	static PVDisplaysImpl& get();
	static void release();

public:
	template <typename F>
	void visit_all_displays(F const& f, PVDisplayIf::Flags flags = PVDisplayIf::NoFlags) const
	{
		visit_displays_by_if<PVDisplayViewIf>(f, flags);
		visit_displays_by_if<PVDisplaySourceIf>(f, flags);
		visit_displays_by_if<PVDisplayViewAxisIf>(f, flags);
	}

	template <typename If, typename F>
	void visit_displays_by_if(F const& f, PVDisplayIf::Flags flags = PVDisplayIf::NoFlags) const
	{
		// Interface of 'F' must be void f(If& obj), or with a base of If;
		typename PVCore::PVClassLibrary<If>::list_classes const& lc = PVCore::PVClassLibrary<If>::get().get_list();
		// `lc' is of type QHash<QString,shared_pointer<If>>
		for (auto it = lc.begin(); it != lc.end(); it++) {
			If& obj = *(it.value());
			if (obj.match_flags(flags)) {
				f(obj);
			}
		}
	}

	template <typename If, typename... P>
	QWidget* get_widget(If& interface, P && ... args) const
	{
		if (interface.match_flags(PVDisplayIf::UniquePerParameters)) {
			return interface.get_unique_widget(std::forward<P>(args)...);
		}

		return interface.create_widget(std::forward<P>(args)...);
	}

private:
	void static_init();
	void static_release();
	void load_plugins();

private:
	static PVDisplaysImpl* _instance;
};


inline PVDisplaysImpl& get() { return PVDisplaysImpl::get(); }
inline void release() { PVDisplaysImpl::release(); }

}

#endif
