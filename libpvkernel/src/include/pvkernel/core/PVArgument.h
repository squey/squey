/**
 * \file PVArgument.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVCORE_PVARGUMENT_H
#define PVCORE_PVARGUMENT_H

#include <pvkernel/core/general.h>
#include <QHash>
#include <QString>
#include <QDomElement>
#include <QVariant>

/*!
 */

namespace PVCore {

/*! \brief PVArgument key that can be used as a QHash key.
 *
 * See \ref PVArgument.h documentation for a complete description of the argument system.
 *
 * \todo The association between a key and its description uses a non thread-safe QHash. For now, this is not an issue,
 * but could become in a close futur.
 * \todo We should be able to create std::map<PVArgumentKey, PVArgument> objects, or any other containers that uses
 * comparaison operations. Thus, it just means "implement operator<" :)
 */
class LibKernelDecl PVArgumentKey: public QString
{
public:
	PVArgumentKey(QString const& key, QString const& desc = QString()):
		QString(key),
		_desc(desc)
	{
		if (desc.isNull()) {
			set_desc_from_key();
		}
		else {
			_key_desc[key] = desc;
		}
	}
	PVArgumentKey(const char* key):
		QString(key)
	{
		set_desc_from_key();
	}

	inline QString const& key() const { return *((QString*)this); }
	inline QString const& desc() const { return _desc; }

private:
	void set_desc_from_key()
	{
		_desc = _key_desc.value(*this, *this);
	}
private:
	QString _desc;
	static QHash<QString, QString> _key_desc;
};


}

extern unsigned int LibKernelDecl qHash(PVCore::PVArgumentKey const& key);

namespace PVCore {

typedef QVariant                           PVArgument;
typedef QHash<PVArgumentKey,PVArgument>    PVArgumentList;
typedef QList<PVArgumentList::key_type>    PVArgumentKeyList;

//class PVArgumentList : public QHash<PVArgumentKey, PVArgument>
//{
//public:
//	int remove(const PVArgumentKey& key);
//	iterator insert(const PVArgumentKey& key, const PVArgument & value);
//
//private:
//	QList<PVArgumentKey> _ordered_keys;
//};

class PVArgumentTypeBase
{
	public:
		PVArgumentTypeBase() {};
		virtual ~PVArgumentTypeBase() {};
	public:
		virtual bool is_equal(const PVArgumentTypeBase &other) const = 0;
		virtual QString to_string() const = 0;
		virtual PVArgument from_string(QString const& str, bool* ok = 0) const = 0;
		virtual void serialize(QDataStream& out) const
		{
			out << to_string();
		}
		virtual PVArgument unserialize(QDataStream& in) const
		{
			QString str;
			in >> str;
			return from_string(str);
		}
};

template <class T>
class PVArgumentType: public PVArgumentTypeBase
{
	virtual bool is_equal(const PVArgumentTypeBase &other) const
	{
		const T* pother = dynamic_cast<const T*>(&other);
		if (!pother) {
			return false;
		}
		return *((T*)this) == *pother;
	}
};


QDataStream &operator<<(QDataStream &out, const PVArgumentTypeBase &obj);
QDataStream &operator>>(QDataStream &in, const PVArgumentTypeBase &obj);

LibKernelDecl QString PVArgument_to_QString(PVArgument const& v);
LibKernelDecl PVArgument QString_to_PVArgument(const QString &s, const QVariant& v, bool* res_ok = 0);

LibKernelDecl void PVArgumentList_to_QSettings(const PVArgumentList& args, QSettings& settings, const QString& group_name);
LibKernelDecl PVArgumentList QSettings_to_PVArgumentList(QSettings& settings, const PVArgumentList& def_args, const QString& group_name);

LibKernelDecl void PVArgumentList_to_QDomElement(const PVArgumentList& args, QDomElement& elt);
LibKernelDecl PVArgumentList QDomElement_to_PVArgumentList(QDomElement const& elt, const PVArgumentList& def_args);

LibKernelDecl void dump_argument_list(PVArgumentList const& l);

LibKernelDecl PVCore::PVArgumentList filter_argument_list_with_keys(PVArgumentList const& args, PVArgumentKeyList const& keys, PVArgumentList const& def_args);

void PVArgumentList_set_common_args_from(PVCore::PVArgumentList& ret, PVCore::PVArgumentList const& ref);
void PVArgumentList_set_missing_args(PVCore::PVArgumentList& ret, PVCore::PVArgumentList const& def_args);

}


#endif
