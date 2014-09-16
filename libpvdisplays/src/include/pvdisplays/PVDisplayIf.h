/**
 * \file PVDisplaysIf.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVDISPLAYS_PVDISPLAYIF_H
#define PVDISPLAYS_PVDISPLAYIF_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVRegistrableClass.h>

#include <picviz/PVSource_types.h>
#include <picviz/PVView_types.h>

#include <QAction>
#include <QIcon>
#include <QMetaType>
#include <QWidget>

#include <unordered_map>

namespace PVDisplays {

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
	PVDisplayIf(int flags = 0, QString const& tooltip_str = QString(), Qt::DockWidgetArea def_pos = Qt::NoDockWidgetArea):
		_flags(flags),
		_tooltip_str(tooltip_str),
		_default_pos(def_pos)
	{ }

	virtual ~PVDisplayIf() { }

public:
	// This function will be called once at the initialisation of the application.
	virtual void static_init() const { }

	// This function will be called once before the end of the application
	virtual void static_release() const { }

public:
	inline int flags() const { return _flags; }
	inline bool match_flags(int f) const { return (flags() & f) == f; }

	inline QString const& tooltip_str() const { return _tooltip_str; }
	inline Qt::DockWidgetArea default_position_hint() const { return _default_pos; }
	inline bool default_position_as_central_hint() const { return default_position_hint() == Qt::NoDockWidgetArea && match_flags(DefaultPresenceInSourceWorkspace); }

public:
	virtual QIcon toolbar_icon() const { return QIcon(); }
	
private:
	int _flags;
	QString _tooltip_str;
	// When set to Qt::NoDockWidgetArea with the DefaultPresenceInSourceWorkspace flag on, it means that we want this widget as a central widget
	Qt::DockWidgetArea _default_pos;
};

template <class T>
class PVDisplayDataTreeIf: public PVDisplayIf
{
	friend class PVDisplaysImpl;

	typedef typename std::remove_pointer<T>::type value_type;
	typedef std::unordered_map<value_type*, QWidget*> hash_widgets_t;

public:
	PVDisplayDataTreeIf(int flags = 0, QString const& tooltip_str = QString(), Qt::DockWidgetArea def_pos = Qt::NoDockWidgetArea):
		PVDisplayIf(flags, tooltip_str, def_pos)
	{ }

public:
	virtual QString widget_title(value_type* /*obj*/) const { return QString(); }

protected:
	QWidget* get_unique_widget(value_type* obj, QWidget* parent = NULL)
	{
		QWidget* ret;
		typename hash_widgets_t::const_iterator it = _widgets.find(obj);
		if (it == _widgets.end()) {
			ret = create_widget(obj, parent);
			_widgets[obj] = ret;
		}
		else {
			ret = it->second;
		}

		return ret;
	}

	QWidget* get_unique_widget_from_action(QAction const& action, QWidget* parent = NULL)
	{
		value_type* obj = get_value_from_action(action);
		return obj == nullptr ? nullptr : get_unique_widget(obj, parent);
	}

	QWidget* create_widget_from_action(QAction const& action, QWidget* parent = NULL) const
	{
		value_type* obj = get_value_from_action(action);
		return obj == nullptr ? nullptr : create_widget(obj, parent);
	}

	inline void get_params_from_action(QAction const& action, value_type* &ret)
	{
		ret = get_value_from_action(action);
	}

	QAction* action_bound_to_params(value_type* obj, QObject* parent = NULL) const
	{
		QAction* action = new QAction(parent);

		QVariant var;
		var.setValue<void*>(reinterpret_cast<void*>(obj));
		action->setData(var);

		return action;
	}

protected:
	virtual QWidget* create_widget(value_type* obj, QWidget* parent = NULL) const = 0;

private:
	inline static value_type* get_value_from_action(QAction const& action)
	{
		return reinterpret_cast<value_type*>(action.data().value<void*>());
	}

private:
	hash_widgets_t _widgets;
};

class PVDisplayViewIf: public PVDisplayDataTreeIf<Picviz::PVView>, public PVCore::PVRegistrableClass<PVDisplayViewIf>
{
public:
	typedef PVDisplayViewIf RegAs;
	typedef std::shared_ptr<RegAs> p_type;

public:
	PVDisplayViewIf(int flags = 0, QString const& tooltip_str = QString(), Qt::DockWidgetArea def_pos = Qt::TopDockWidgetArea):
		PVDisplayDataTreeIf<Picviz::PVView>(flags, tooltip_str, def_pos)
	{ }
};

class PVDisplaySourceIf: public PVDisplayDataTreeIf<Picviz::PVSource>, public PVCore::PVRegistrableClass<PVDisplaySourceIf>
{
public:
	typedef PVDisplaySourceIf RegAs;
	typedef std::shared_ptr<RegAs> p_type;

public:
	PVDisplaySourceIf(int flags = 0, QString const& tooltip_str = QString(), Qt::DockWidgetArea def_pos = Qt::TopDockWidgetArea):
		PVDisplayDataTreeIf<Picviz::PVSource>(flags, tooltip_str, def_pos)
	{ }
};

namespace __impl {

class PVDisplayViewAxisIf: public PVDisplayIf
{
public:
	struct Params
	{
		Params(): view(nullptr), axis_comb(0) { }
		Params(const Params& o): view(o.view), axis_comb(o.axis_comb) { }
		Params(Picviz::PVView* view_, PVCol axis_comb_):
			view(view_),
			axis_comb(axis_comb_)
		{ }

		Picviz::PVView* view;
		PVCol axis_comb;

		inline bool operator<(Params const& p) const { return view < p.view && axis_comb < p.axis_comb; }
	};

private:
	typedef std::map<Params, QWidget*> map_widgets_t;

public:
	PVDisplayViewAxisIf(int flags = 0, QString const& tooltip_str = QString(), Qt::DockWidgetArea def_pos = Qt::TopDockWidgetArea):
		PVDisplayIf(flags, tooltip_str, def_pos)
	{ }

public:
	virtual QString widget_title(Picviz::PVView* /*obj*/, PVCol /*axis_comb*/) const { return QString(); }
	virtual QString axis_menu_name(Picviz::PVView const* /*obj*/, PVCol /*axis_comb*/) const { return QString(); }

protected:
	QWidget* get_unique_widget(Picviz::PVView* view, PVCol axis_comb, QWidget* parent = NULL);
	QWidget* get_unique_widget_from_action(QAction const& action, QWidget* parent = NULL);

	QWidget* create_widget_from_action(QAction const& action, QWidget* parent = NULL) const;

	inline void get_params_from_action(QAction const& action, Picviz::PVView* &view, PVCol& axis_comb)
	{
		Params p = get_params_from_action(action);
		view = p.view;
		axis_comb = p.axis_comb;
	}

	QAction* action_bound_to_params(Picviz::PVView* view, PVCol axis_comb, QObject* parent = NULL) const;

protected:
	virtual QWidget* create_widget(Picviz::PVView* view, PVCol axis_comb, QWidget* parent = NULL) const = 0;

private:
	inline static Params get_params_from_action(QAction const& action)
	{
		return action.data().value<Params>();
	}

private:
	map_widgets_t _widgets;
};

}

class PVDisplayViewAxisIf: public __impl::PVDisplayViewAxisIf, public PVCore::PVRegistrableClass<PVDisplayViewAxisIf>
{
	friend class PVDisplaysImpl;

public:
	PVDisplayViewAxisIf(int flags = 0, QString const& tooltip_str = QString(), Qt::DockWidgetArea def_pos = Qt::TopDockWidgetArea):
		__impl::PVDisplayViewAxisIf(flags, tooltip_str, def_pos)
	{ }

public:
	typedef PVDisplayViewAxisIf RegAs;
	typedef std::shared_ptr<RegAs> p_type;
};

class PVDisplayViewZoneIf: public __impl::PVDisplayViewAxisIf, public PVCore::PVRegistrableClass<PVDisplayViewZoneIf>
{
	friend class PVDisplaysImpl;

public:
	PVDisplayViewZoneIf(int flags = 0, QString const& tooltip_str = QString(), Qt::DockWidgetArea def_pos = Qt::TopDockWidgetArea):
		__impl::PVDisplayViewAxisIf(flags, tooltip_str, def_pos)
	{ }

public:
	typedef PVDisplayViewZoneIf RegAs;
	typedef std::shared_ptr<RegAs> p_type;
};

}

Q_DECLARE_METATYPE(PVDisplays::PVDisplayViewAxisIf::Params)

#endif
