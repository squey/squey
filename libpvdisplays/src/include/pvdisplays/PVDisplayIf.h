#ifndef PVDISPLAYS_PVDISPLAYIF_H
#define PVDISPLAYS_PVDISPLAYIF_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVRegistrableClass.h>

#include <picviz/PVSource_types.h>
#include <picviz/PVView_types.h>

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
		ShowInCentralDockWidget = 8
	} Flags;

protected:
	PVDisplayIf(int flags = 0):
		_flags(flags)
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
	
private:
	int _flags;
};

template <class T>
class PVDisplayDataTreeIf: public PVDisplayIf
{
	friend class PVDisplaysImpl;

	typedef typename std::remove_pointer<T>::type value_type;
	typedef std::unordered_map<value_type*, QWidget*> hash_widgets_t;

public:
	PVDisplayDataTreeIf(int flags = 0):
		PVDisplayIf(flags)
	{ }

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
			assert(ret->parent() == parent);
		}

		return ret;
	}

protected:
	virtual QWidget* create_widget(value_type* obj, QWidget* parent = NULL) const = 0;

private:
	hash_widgets_t _widgets;
};

class PVDisplayViewIf: public PVDisplayDataTreeIf<Picviz::PVView>, public PVCore::PVRegistrableClass<PVDisplayViewIf>
{
public:
	typedef PVDisplayViewIf RegAs;
	typedef boost::shared_ptr<RegAs> p_type;

public:
	PVDisplayViewIf(int flags = 0):
		PVDisplayDataTreeIf<Picviz::PVView>(flags)
	{ }
};

class PVDisplaySourceIf: public PVDisplayDataTreeIf<Picviz::PVSource>, public PVCore::PVRegistrableClass<PVDisplaySourceIf>
{
public:
	typedef PVDisplaySourceIf RegAs;
	typedef boost::shared_ptr<RegAs> p_type;

public:
	PVDisplaySourceIf(int flags = 0):
		PVDisplayDataTreeIf<Picviz::PVSource>(flags)
	{ }
};

class PVDisplayViewAxisIf: public PVDisplayIf, public PVCore::PVRegistrableClass<PVDisplayViewAxisIf>
{
	friend class PVDisplaysImpl;

	struct Params
	{
		Params(Picviz::PVView* view_, PVCol axis_comb_):
			view(view_),
			axis_comb(axis_comb_)
		{ }

		Picviz::PVView* view;
		PVCol axis_comb;

		inline bool operator<(Params const& p) const { return view < p.view && axis_comb < p.axis_comb; }
	};

	typedef std::map<Params, QWidget*> map_widgets_t;

public:
	typedef PVDisplayViewAxisIf RegAs;
	typedef boost::shared_ptr<RegAs> p_type;

public:
	PVDisplayViewAxisIf(int flags = 0):
		PVDisplayIf(flags)
	{ }

protected:
	QWidget* get_unique_widget(Picviz::PVView* view, PVCol axis_comb, QWidget* parent = NULL);

protected:
	virtual QWidget* create_widget(Picviz::PVView* view, PVCol axis_comb, QWidget* parent = NULL) const = 0;

private:
	map_widgets_t _widgets;
};

}

#endif
