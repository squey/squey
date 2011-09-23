#ifndef PVCORE_PVTAG_FILE_H
#define PVCORE_PVTAG_FILE_H

#include <pvkernel/core/general.h>

#include <QList>
#include <QString>
#include <exception>

namespace PVCore {

template<class RegAs>
class PVClassLibrary;

class PVTagUndefinedException
{
public:
	PVTagUndefinedException(QString const& tag_name):
		_tag_name(tag_name)
	{ }
public:
	QString what() { return QString("Tag %1 undefined").arg(_tag_name); }
protected:
	QString _tag_name;
};

template <class TReg>
class LibKernelDecl PVTag
{
	template<class RegAs>
	friend class PVClassLibrary;
public:
	typedef typename TReg::p_type PF;
	typedef QList<PF> list_classes;

protected:
	PVTag(QString const& name):
		_name(name)
	{ }

public:
	QString const& name() const { return _name; }
	list_classes const& associated_classes() { return _associated_classes; }

public:
	operator QString() const { return _name; }
	bool operator==(PVTag const& tag) const { return _name == tag._name; }

protected:
	void add_class(PF p) { _associated_classes.push_back(p); }

protected:
	QString _name;
	list_classes _associated_classes;
};

}

#endif
