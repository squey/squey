/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDISPLAYS_PVDISPLAYIF_H
#define PVDISPLAYS_PVDISPLAYIF_H

#include <pvbase/types.h> // for PVCombCol

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVRegistrableClass.h>

#include <pvdisplays/PVDisplaysContainer.h>

#include <QMenu>
#include <QAction>
#include <QIcon>
#include <QMetaType>
#include <QWidget>
#include <QString>
#include <QDebug>

#include <unordered_map>
#include <vector>
#include <any>

namespace Inendi
{
class PVSource;
class PVView;
} // namespace Inendi

namespace PVDisplays
{
class PVDisplaysContainer;

class PVDisplayIf
{
  public:
	typedef enum {
		UniquePerParameters = 1,
		ShowInToolbar = 2,
		ShowInDockWidget = 4,
		ShowInCentralDockWidget = 8,
		ShowInCtxtMenu = 16,
		DefaultPresenceInSourceWorkspace = 32
	} Flags;

  protected:
	explicit PVDisplayIf(int flags = 0,
	                     QString tooltip_str = QString(),
	                     Qt::DockWidgetArea def_pos = Qt::NoDockWidgetArea)
	    : _flags(flags), _tooltip_str(std::move(tooltip_str)), _default_pos(def_pos)
	{
	}

	virtual ~PVDisplayIf() = default;

  public:
	inline int flags() const { return _flags; }
	inline bool match_flags(int f) const { return (flags() & f) == f; }

	inline QString const& tooltip_str() const { return _tooltip_str; }
	inline Qt::DockWidgetArea default_position_hint() const { return _default_pos; }
	inline bool default_position_as_central_hint() const
	{
		return default_position_hint() == Qt::NoDockWidgetArea &&
		       match_flags(DefaultPresenceInSourceWorkspace);
	}

  public:
	virtual QIcon toolbar_icon() const { return QIcon(); }

  private:
	int _flags;
	QString _tooltip_str;
	// When set to Qt::NoDockWidgetArea with the DefaultPresenceInSourceWorkspace flag on, it means
	// that we want this widget as a central widget
	Qt::DockWidgetArea _default_pos;
};

template <class T>
class PVDisplayDataTreeIf : public PVDisplayIf
{
	using value_type = std::remove_pointer_t<T>;
	using hash_widgets_t = std::unordered_map<value_type*, QWidget*>;

  public:
	using Params = std::vector<std::any>;

	using PVDisplayIf::PVDisplayIf;

	virtual QString widget_title(value_type* /*obj*/) const { return QString(); }

	QWidget* get_unique_widget(value_type* obj, QWidget* parent = nullptr, Params const& data = {})
	{
		if (auto it = _widgets.find(obj); it != _widgets.end()) {
			return it->second;
		}
		return _widgets[obj] = create_widget(obj, parent, data);
	}

	virtual QWidget*
	create_widget(value_type* obj, QWidget* parent = nullptr, Params const& data = {}) const = 0;

  private:
	hash_widgets_t _widgets;
};

class PVDisplayViewIf : public PVDisplayDataTreeIf<Inendi::PVView>,
                        public PVCore::PVRegistrableClass<PVDisplayViewIf>
{
  public:
	using RegAs = PVDisplayViewIf;
	using p_type = std::shared_ptr<RegAs>;

	using PVDisplayDataTreeIf::PVDisplayDataTreeIf;

	virtual QString axis_menu_name(Inendi::PVView*) const { return QString(); }
	virtual void add_to_axis_menu(QMenu& menu,
	                              PVCol axis,
	                              PVCombCol axis_comb,
	                              Inendi::PVView* view,
	                              PVDisplaysContainer* container);
};

class PVDisplaySourceIf : public PVDisplayDataTreeIf<Inendi::PVSource>,
                          public PVCore::PVRegistrableClass<PVDisplaySourceIf>
{
  public:
	using RegAs = PVDisplaySourceIf;
	using p_type = std::shared_ptr<RegAs>;

	using PVDisplayDataTreeIf::PVDisplayDataTreeIf;
};

/******************************
 * Helper functions
 * ****************************/

template <typename If, typename F>
void visit_displays_by_if(F const& f, int flags = 0)
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

template <class If>
If& display_view_if()
{
	If* display = nullptr;
	visit_displays_by_if<PVDisplayViewIf>([&display](auto& obj) {
		if (display == nullptr) {
			display = dynamic_cast<If*>(&obj);
		}
	});
	assert(display != nullptr);
	return *display;
}

template <typename If, typename... P>
QWidget* get_widget(If& interface, P&&... args)
{
	if (interface.match_flags(PVDisplayIf::UniquePerParameters)) {
		return interface.get_unique_widget(std::forward<P>(args)...);
	}

	return interface.create_widget(std::forward<P>(args)...);
}

/**
 * @param axis is always valid
 * @param axis_comb may be empty
 **/
void add_displays_view_axis_menu(QMenu& menu,
                                 PVDisplaysContainer* container,
                                 Inendi::PVView* view,
                                 PVCol axis,
                                 PVCombCol axis_comb = PVCombCol());

/**
 * @param axis_comb is used to compute PVCol axis
 **/
void add_displays_view_axis_menu(QMenu& menu,
                                 PVDisplaysContainer* container,
                                 Inendi::PVView* view,
                                 PVCombCol axis_comb);

PVCol col_param(Inendi::PVView* view, std::vector<std::any> const& params, size_t index);

} // namespace PVDisplays

#endif
