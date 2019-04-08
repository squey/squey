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

#include <QAction>
#include <QIcon>
#include <QMetaType>
#include <QWidget>

#include <unordered_map>

namespace Inendi
{
class PVSource;
class PVView;
} // namespace Inendi

namespace PVDisplays
{

class PVDisplaysImpl;

class PVDisplayIf
{
	friend class PVDisplaysImpl;

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
	friend class PVDisplaysImpl;

	typedef typename std::remove_pointer<T>::type value_type;
	typedef std::unordered_map<value_type*, QWidget*> hash_widgets_t;

  public:
	explicit PVDisplayDataTreeIf(int flags = 0,
	                             QString const& tooltip_str = QString(),
	                             Qt::DockWidgetArea def_pos = Qt::NoDockWidgetArea)
	    : PVDisplayIf(flags, tooltip_str, def_pos)
	{
	}

  public:
	virtual QString widget_title(value_type* /*obj*/) const { return QString(); }

  protected:
	QWidget* get_unique_widget(value_type* obj, QWidget* parent = nullptr)
	{
		if (auto it = _widgets.find(obj); it != _widgets.end()) {
			return it->second;
		}
		return _widgets[obj] = create_widget(obj, parent);
	}

  protected:
	virtual QWidget* create_widget(value_type* obj, QWidget* parent = nullptr) const = 0;

  private:
	hash_widgets_t _widgets;
};

class PVDisplayViewIf : public PVDisplayDataTreeIf<Inendi::PVView>,
                        public PVCore::PVRegistrableClass<PVDisplayViewIf>
{
  public:
	typedef PVDisplayViewIf RegAs;
	typedef std::shared_ptr<RegAs> p_type;

  public:
	explicit PVDisplayViewIf(int flags = 0,
	                         QString const& tooltip_str = QString(),
	                         Qt::DockWidgetArea def_pos = Qt::TopDockWidgetArea)
	    : PVDisplayDataTreeIf<Inendi::PVView>(flags, tooltip_str, def_pos)
	{
	}
};

class PVDisplaySourceIf : public PVDisplayDataTreeIf<Inendi::PVSource>,
                          public PVCore::PVRegistrableClass<PVDisplaySourceIf>
{
  public:
	typedef PVDisplaySourceIf RegAs;
	typedef std::shared_ptr<RegAs> p_type;

  public:
	explicit PVDisplaySourceIf(int flags = 0,
	                           QString const& tooltip_str = QString(),
	                           Qt::DockWidgetArea def_pos = Qt::TopDockWidgetArea)
	    : PVDisplayDataTreeIf<Inendi::PVSource>(flags, tooltip_str, def_pos)
	{
	}
};

class PVDisplayViewDataIf : public PVDisplayIf,
                            public PVCore::PVRegistrableClass<PVDisplayViewDataIf>
{
	friend class PVDisplaysImpl;

  public:
	using Params = std::vector<PVCombCol>;

  private:
	using map_widgets_t = QMap<Inendi::PVView*, QWidget*>;

  public:
	explicit PVDisplayViewDataIf(int flags = 0,
	                             QString const& tooltip_str = QString(),
	                             Qt::DockWidgetArea def_pos = Qt::TopDockWidgetArea)
	    : PVDisplayIf(flags, tooltip_str, def_pos)
	{
	}

  public:
	virtual QString widget_title(Inendi::PVView*, Params const&) const { return QString(); }
	virtual QString axis_menu_name(Inendi::PVView*, Params const&) const { return QString(); }
	virtual bool should_add_to_menu(Inendi::PVView*, Params const&) const { return true; }

  protected:
	QWidget* get_unique_widget(Inendi::PVView* view, Params const& data, QWidget* parent = nullptr)
	{
		if (auto it = _widgets.find(view); it != _widgets.end()) {
			return it.value();
		}
		return _widgets[view] = create_widget(view, data, parent);
	}

  protected:
	virtual QWidget*
	create_widget(Inendi::PVView* view, Params const& data, QWidget* parent = nullptr) const = 0;

  private:
	map_widgets_t _widgets;

  public:
	using RegAs = PVDisplayViewDataIf;
	using p_type = std::shared_ptr<RegAs>;
};

using ViewAxisParams = std::tuple<Inendi::PVView*, PVCombCol>;
using ViewZoneParams = std::tuple<Inendi::PVView*, PVCombCol, PVCombCol>;

class PVDisplayViewZoneIf : public PVDisplayIf,
                            public PVCore::PVRegistrableClass<PVDisplayViewZoneIf>
{
	friend class PVDisplaysImpl;

  public:
	struct Params {
		Params() : view(nullptr), axis_comb_first(0), axis_comb_second(0) {}
		Params(const Params& o) = default;
		Params(Inendi::PVView* view_, PVCombCol axis_comb_first_, PVCombCol axis_comb_second_)
		    : view(view_), axis_comb_first(axis_comb_first_), axis_comb_second(axis_comb_second_)
		{
		}

		Inendi::PVView* view;
		PVCombCol axis_comb_first;
		PVCombCol axis_comb_second;
		bool ask_for_box;

		inline bool operator<(Params const& p) const
		{
			return view < p.view && axis_comb_first < p.axis_comb_first &&
			       axis_comb_second < p.axis_comb_second;
		}
	};

  private:
	typedef std::map<Params, QWidget*> map_widgets_t;

  public:
	explicit PVDisplayViewZoneIf(int flags = 0,
	                             QString const& tooltip_str = QString(),
	                             Qt::DockWidgetArea def_pos = Qt::TopDockWidgetArea)
	    : PVDisplayIf(flags, tooltip_str, def_pos)
	{
	}

  public:
	virtual QString widget_title(Inendi::PVView* /*obj*/,
	                             PVCombCol /*axis_comb_first*/,
	                             PVCombCol /*axis_comb_second*/) const
	{
		return QString();
	}
	virtual QString axis_menu_name(Inendi::PVView const* /*obj*/,
	                               PVCombCol /*axis_comb_first*/,
	                               PVCombCol /*axis_comb_second*/) const
	{
		return QString();
	}

  protected:
	QWidget* get_unique_widget(Inendi::PVView* view,
	                           PVCombCol axis_comb_first,
	                           PVCombCol axis_comb_second,
	                           QWidget* parent = nullptr)
	{
		QWidget* ret;
		map_widgets_t::const_iterator it =
		    _widgets.find(Params(view, axis_comb_first, axis_comb_second));

		if (it == _widgets.end()) {
			ret = create_widget(view, axis_comb_first, axis_comb_second, parent);
			_widgets[Params(view, axis_comb_first, axis_comb_second)] = ret;
		} else {
			ret = it->second;
			assert(ret->parent() == parent);
		}

		return ret;
	}

  protected:
	virtual QWidget* create_widget(Inendi::PVView* view,
	                               PVCombCol axis_comb_first,
	                               PVCombCol axis_comb_second,
	                               QWidget* parent = nullptr) const = 0;

  private:
	map_widgets_t _widgets;

  public:
	typedef PVDisplayViewZoneIf RegAs;
	typedef std::shared_ptr<RegAs> p_type;
};
} // namespace PVDisplays

Q_DECLARE_METATYPE(PVDisplays::ViewAxisParams)
Q_DECLARE_METATYPE(PVDisplays::ViewZoneParams)
Q_DECLARE_METATYPE(PVDisplays::PVDisplayViewDataIf::Params)
Q_DECLARE_METATYPE(PVDisplays::PVDisplayViewZoneIf::Params)

#endif
