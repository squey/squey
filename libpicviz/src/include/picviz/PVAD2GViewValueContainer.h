#ifndef PICVIZ_PVAD2GVIEWVALUECONTAINER_H
#define PICVIZ_PVAD2GVIEWVALUECONTAINER_H

#include <picviz/PVCombiningFunctionView.h>

namespace Picviz {

template<class T>
class PVAD2GViewValueContainer
{
public:
	PVAD2GViewValueContainer(int i = 0) { (void)i; }
	PVAD2GViewValueContainer(T data) { set_data(data); }
	~PVAD2GViewValueContainer() {}

public:
	void set_data(T data) { _data = data; }
	T get_data() const { return _data; }

public:
	bool operator != (const PVAD2GViewValueContainer<T> &c) const { return _data != c._data; }
	bool operator == (const PVAD2GViewValueContainer<T> &c) const { return _data == c._data; }
	bool operator <  (const PVAD2GViewValueContainer<T> &c) const { return _data < c._data; }

private:
	T _data;
};

}

#endif // PICVIZ_PVAD2GVIEWVALUECONTAINER_H
