#ifndef PICVIZ_PVAD2GVIEWPOINTERCONTAINER_H
#define PICVIZ_PVAD2GVIEWPOINTERCONTAINER_H

#include <picviz/PVView.h>
#include <picviz/PVCombiningFunctionView.h>

#include <boost/shared_ptr.hpp>

namespace Picviz {

template<class T>
class PVAD2GViewPointerContainer
{
public:
	PVAD2GViewPointerContainer(T *data = 0) { set_data(data); }
	~PVAD2GViewPointerContainer() { _data = 0; }

public:
	void set_data(T *data) { _data = data; }
	T *get_data() const { return _data; }

public:
	bool operator != (const PVAD2GViewPointerContainer<T> &c) const { return _data != c._data; }
	bool operator == (const PVAD2GViewPointerContainer<T> &c) const { return _data == c._data; }
	bool operator <  (const PVAD2GViewPointerContainer<T> &c) const { return _data < c._data; }

private:
	T *_data;
};

typedef PVAD2GViewPointerContainer<Picviz::PVView*> PVAD2GViewNode;

typedef boost::shared_ptr<PVCombiningFunctionView> PVCombiningFunctionView_p;

typedef PVAD2GViewPointerContainer<PVCombiningFunctionView_p> PVAD2GViewEdge;

}

#endif // PICVIZ_PVAD2GVIEWPOINTERCONTAINER_H
