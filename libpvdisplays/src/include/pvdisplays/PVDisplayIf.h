/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVDISPLAYS_PVDISPLAYIF_H
#define PVDISPLAYS_PVDISPLAYIF_H

#include <inendi/PVCombCol.h>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVRegistrableClass.h>

#include <pvbase/types.h>

#include <QAction>
#include <QIcon>
#include <QMetaType>
#include <QWidget>

#include <unordered_map>

namespace Inendi
{
class PVSource;
class PVView;
}

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
	PVDisplayIf(int flags = 0,
	            QString const& tooltip_str = QString(),
	            Qt::DockWidgetArea def_pos = Qt::NoDockWidgetArea)
	    : _flags(flags), _tooltip_str(tooltip_str), _default_pos(def_pos)
	{
	}

	virtual ~PVDisplayIf() {}

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
	PVDisplayDataTreeIf(int flags = 0,
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
		QWidget* ret;
		typename hash_widgets_t::const_iterator it = _widgets.find(obj);
		if (it == _widgets.end()) {
			ret = create_widget(obj, parent);
			_widgets[obj] = ret;
		} else {
			ret = it->second;
		}

		return ret;
	}

	inline void get_params_from_action(QAction const& action, value_type*& ret)
	{
		ret = get_value_from_action(action);
	}

	QAction* action_bound_to_params(value_type* obj,
	                                Inendi::PVCombCol /*axis_comb*/,
	                                QObject* parent = nullptr) const
	{
		QAction* action = new QAction(parent);

		QVariant var;
		var.setValue<void*>(reinterpret_cast<void*>(obj));
		action->setData(var);

		return action;
	}

  protected:
	virtual QWidget* create_widget(value_type* obj, QWidget* parent = nullptr) const = 0;

  private:
	inline static value_type* get_value_from_action(QAction const& action)
	{
		return reinterpret_cast<value_type*>(action.data().value<void*>());
	}

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
	PVDisplayViewIf(int flags = 0,
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
	PVDisplaySourceIf(int flags = 0,
	                  QString const& tooltip_str = QString(),
	                  Qt::DockWidgetArea def_pos = Qt::TopDockWidgetArea)
	    : PVDisplayDataTreeIf<Inendi::PVSource>(flags, tooltip_str, def_pos)
	{
	}
};

namespace __impl
{

class PVDisplayViewAxisIf : public PVDisplayIf
{
  public:
	struct Params {
		Params() : view(nullptr), axis_comb(0) {}
		Params(const Params& o) : view(o.view), axis_comb(o.axis_comb) {}
		Params(Inendi::PVView* view_, Inendi::PVCombCol axis_comb_)
		    : view(view_), axis_comb(axis_comb_)
		{
		}

		Inendi::PVView* view;
		Inendi::PVCombCol axis_comb;

		inline bool operator<(Params const& p) const
		{
			return view < p.view && axis_comb < p.axis_comb;
		}
	};

  private:
	typedef std::map<Params, QWidget*> map_widgets_t;

  public:
	PVDisplayViewAxisIf(int flags = 0,
	                    QString const& tooltip_str = QString(),
	                    Qt::DockWidgetArea def_pos = Qt::TopDockWidgetArea)
	    : PVDisplayIf(flags, tooltip_str, def_pos)
	{
	}

  public:
	virtual QString widget_title(Inendi::PVView* /*obj*/, Inendi::PVCombCol /*axis_comb*/) const
	{
		return QString();
	}
	virtual QString axis_menu_name(Inendi::PVView const* /*obj*/,
	                               Inendi::PVCombCol /*axis_comb*/) const
	{
		return QString();
	}

  protected:
	QWidget*
	get_unique_widget(Inendi::PVView* view, Inendi::PVCombCol axis_comb, QWidget* parent = nullptr);

	inline void get_params_from_action(QAction const& action,
	                                   Inendi::PVView*& view,
	                                   Inendi::PVCombCol& axis_comb)
	{
		Params p = get_params_from_action(action);
		view = p.view;
		axis_comb = p.axis_comb;
	}

	QAction* action_bound_to_params(Inendi::PVView* view,
	                                Inendi::PVCombCol axis_comb,
	                                QObject* parent = nullptr) const;

  protected:
	virtual QWidget* create_widget(Inendi::PVView* view,
	                               Inendi::PVCombCol axis_comb,
	                               QWidget* parent = nullptr) const = 0;

  private:
	inline static Params get_params_from_action(QAction const& action)
	{
		return action.data().value<Params>();
	}

  private:
	map_widgets_t _widgets;
};
}

class PVDisplayViewAxisIf : public __impl::PVDisplayViewAxisIf,
                            public PVCore::PVRegistrableClass<PVDisplayViewAxisIf>
{
	friend class PVDisplaysImpl;

  public:
	PVDisplayViewAxisIf(int flags = 0,
	                    QString const& tooltip_str = QString(),
	                    Qt::DockWidgetArea def_pos = Qt::TopDockWidgetArea)
	    : __impl::PVDisplayViewAxisIf(flags, tooltip_str, def_pos)
	{
	}

  public:
	typedef PVDisplayViewAxisIf RegAs;
	typedef std::shared_ptr<RegAs> p_type;
};

class PVDisplayViewZoneIf : public __impl::PVDisplayViewAxisIf,
                            public PVCore::PVRegistrableClass<PVDisplayViewZoneIf>
{
	friend class PVDisplaysImpl;

  public:
	PVDisplayViewZoneIf(int flags = 0,
	                    QString const& tooltip_str = QString(),
	                    Qt::DockWidgetArea def_pos = Qt::TopDockWidgetArea)
	    : __impl::PVDisplayViewAxisIf(flags, tooltip_str, def_pos)
	{
	}

  public:
	typedef PVDisplayViewZoneIf RegAs;
	typedef std::shared_ptr<RegAs> p_type;
};
}

Q_DECLARE_METATYPE(PVDisplays::PVDisplayViewAxisIf::Params)

#endif
