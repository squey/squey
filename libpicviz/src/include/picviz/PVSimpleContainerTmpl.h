#ifndef PICVIZ_PVSIMPLECONTAINERTMPL_H
#define PICVIZ_PVSIMPLECONTAINERTMPL_H

namespace Picviz {

/* This template is principaly used by PVAD2GView to store informations in
 * the Tulip graph.
 */
template<class T>
class PVSimpleContainerTmpl
{
public:
	PVSimpleContainerTmpl(int i = 0) { (void)i; }
	PVSimpleContainerTmpl(T data) { set_data(data); }
	~PVSimpleContainerTmpl() {}

public:
	void set_data(T data) { _data = data; }
	T get_data() const { return _data; }

public:
	bool operator != (const PVSimpleContainerTmpl<T> &c) const { return _data != c._data; }
	bool operator == (const PVSimpleContainerTmpl<T> &c) const { return _data == c._data; }
	bool operator <  (const PVSimpleContainerTmpl<T> &c) const { return _data < c._data; }

private:
	T _data;
};

}

#endif // PICVIZ_PVSIMPLECONTAINERTMPL_H
