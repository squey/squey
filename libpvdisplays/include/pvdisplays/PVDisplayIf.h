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

namespace Squey
{
class PVSource;
class PVView;
} // namespace Squey

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
	                     QString tooltip_str = {},
	                     QIcon toolbar_icon = {},
	                     Qt::DockWidgetArea def_pos = Qt::NoDockWidgetArea)
	    : _flags(flags)
	    , _tooltip_str(std::move(tooltip_str))
	    , _default_pos(def_pos)
	    , _toolbar_icon(toolbar_icon)
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

	QIcon toolbar_icon() const { return _toolbar_icon; }

  private:
	int _flags;
	QString _tooltip_str;
	// When set to Qt::NoDockWidgetArea with the DefaultPresenceInSourceWorkspace flag on, it means
	// that we want this widget as a central widget
	Qt::DockWidgetArea _default_pos;
	QIcon _toolbar_icon;
};

template <class T>
class PVDisplayDataTreeIf : public PVDisplayIf
{
	using value_type = std::remove_pointer_t<T>;
	using hash_widgets_t = std::unordered_map<value_type*, QWidget*>;

  public:
	using Params = std::vector<std::any>;

	using PVDisplayIf::PVDisplayIf;

	QWidget* get_unique_widget(value_type* obj, QWidget* parent = nullptr, Params const& data = {})
	{
		if (auto it = _widgets.find(obj); it != _widgets.end()) {
			if (auto parent = it->second->parentWidget()) {
				parent->deleteLater();
			}
			return it->second;
		}
		auto w = _widgets[obj] = create_widget(obj, parent, data);
		w->connect(w, &QWidget::destroyed, [obj, this](QObject*) { _widgets.erase(obj); });
		return w;
	}

	virtual QWidget*
	create_widget(value_type* obj, QWidget* parent = nullptr, Params const& data = {}) const = 0;

  private:
	hash_widgets_t _widgets;
};

class PVDisplayViewIf : public PVDisplayDataTreeIf<Squey::PVView>,
                        public PVCore::PVRegistrableClass<PVDisplayViewIf>
{
  public:
	using RegAs = PVDisplayViewIf;
	using p_type = std::shared_ptr<RegAs>;

	using PVDisplayDataTreeIf::PVDisplayDataTreeIf;
	PVDisplayViewIf(int flags = 0,
	                QString tooltip_str = {},
	                QIcon toolbar_icon = {},
	                QString axis_menu_name = {},
	                Qt::DockWidgetArea def_pos = Qt::NoDockWidgetArea)
	    : PVDisplayDataTreeIf(flags, tooltip_str, toolbar_icon, def_pos)
	    , _axis_menu_name(axis_menu_name)
	{
	}

	virtual QString axis_menu_name() const { return _axis_menu_name; }
	virtual void add_to_axis_menu(QMenu& menu,
	                              PVCol axis,
	                              PVCombCol axis_comb,
	                              Squey::PVView* view,
	                              PVDisplaysContainer* container);

	/** @brief Can be used with QWidget::setWindowTitle() when no other info is needed */
	QString default_window_title(Squey::PVView& view) const;

  private:
	QString _axis_menu_name;
};

class PVDisplaySourceIf : public PVDisplayDataTreeIf<Squey::PVSource>,
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
                                 Squey::PVView* view,
                                 PVCol axis,
                                 PVCombCol axis_comb = PVCombCol());

/**
 * @param axis_comb is used to compute PVCol axis
 **/
void add_displays_view_axis_menu(QMenu& menu,
                                 PVDisplaysContainer* container,
                                 Squey::PVView* view,
                                 PVCombCol axis_comb);

PVCol col_param(Squey::PVView* view, std::vector<std::any> const& params, size_t index);

} // namespace PVDisplays

#endif
