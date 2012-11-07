#ifndef PVDISPLAYS_PVDISPLAYSIMPL_H
#define PVDISPLAYS_PVDISPLAYSIMPL_H

#include <pvdisplays/PVDisplayIf.h>
#include <QMetaType>

namespace PVDisplays {

class PVDisplaysImpl: public QObject
{
public:
	struct ActionParams
	{
		PVDisplayIf* disp_if;
		QVariant params;
	};

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
	void visit_all_displays(F const& f, int flags = 0) const
	{
		visit_displays_by_if<PVDisplayViewIf>(f, flags);
		visit_displays_by_if<PVDisplaySourceIf>(f, flags);
		visit_displays_by_if<PVDisplayViewAxisIf>(f, flags);
	}

	template <typename If, typename F>
	void visit_displays_by_if(F const& f, int flags = 0) const
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

	/*
	 * AG: fix this!
	QWidget* get_widget_from_action(QAction& action, QWidget* parent = NULL) const
	{
		QVariant org_data = action.data();

		ActionParams p = action.data().value<ActionParams>();
		PVDisplayIf& interface = *p.disp_if;

		action.setData(p.params);

		QWidget* ret;
		if (interface.match_flags(PVDisplayIf::UniquePerParameters)) {
			ret = interface.get_unique_widget_from_action(action, parent);
		}
		else {
			ret = interface.create_widget_from_action(action, parent);
		}

		action.setData(org_data);

		return ret;
	}*/

	template <typename If, typename... P>
	If& get_params_from_action(QAction& action, P && ... args) const
	{
		QVariant org_data = action.data();

		ActionParams p = action.data().value<ActionParams>();
		If* interface = dynamic_cast<If*>(p.disp_if);
		assert(interface);

		action.setData(p.params);
		interface->get_params_from_action(action, std::forward<P>(args)...);
		action.setData(org_data);

		return *interface;
	}

	template <typename If, typename... P>
	inline QAction* action_bound_to_params(If& interface, P && ... args) const
	{
		// Get the action from the interface and add the interface itself as an argument to QAction
		QAction* act = interface.action_bound_to_params(std::forward<P>(args)...);
		ActionParams p;
		p.disp_if = static_cast<PVDisplayIf*>(&interface);
		p.params = act->data();
		
		QVariant var;
		var.setValue<ActionParams>(p);
		act->setData(var);

		return act;
	}

	void add_displays_view_axis_menu(QMenu& menu, QObject* receiver, const char* slot, Picviz::PVView* view, PVCol axis_comb) const;

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

Q_DECLARE_METATYPE(PVDisplays::PVDisplaysImpl::ActionParams)

#endif
