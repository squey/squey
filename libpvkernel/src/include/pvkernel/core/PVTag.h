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

public:
	// Creates an invalid tag
	PVTag() { _valid = false; }

protected:
	// Only PVClassLibrary can create tags, that are thus always valid
	PVTag(QString const& name, QString const& desc):
		_name(name),
		_desc(desc)
	{ _valid = true; }

public:
	QString const& name() const { return _name; }
	QString const& desc() const { return _desc; }
	list_classes const& associated_classes() const { return _associated_classes; }
	bool valid() const { return _valid; }

public:
	operator QString() const { return _name; }
	bool operator==(PVTag const& tag) const
	{
		if (!_valid) {
			return !tag._valid;
		}
		return _name == tag._name;
	}

protected:
	void add_class(PF p) { _associated_classes.push_back(p); }

protected:
	QString _name;
	QString _desc;
	list_classes _associated_classes;
	bool _valid;
};

}

#endif
