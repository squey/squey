/**
 * \file PVArgument.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVTimeFormatType.h>
#include <QStringList>
#include <QRect>
#include <QMetaType>

QHash<QString,QString> PVCore::PVArgumentKey::_key_desc;

// Inspired from QSettingsPrivate functions !

//int PVCore::PVArgumentList::remove(const PVArgumentKey& key)
//{
//
//}
//
//PVCore::PVArgumentList::iterator PVCore::PVArgumentList::insert(const PVArgumentKey& key, const PVArgument & value)
//{
//	QHash::insert()
//}

static QStringList splitArgs(const QString &s, int idx)
{
	int l = s.length();
	Q_ASSERT(l > 0);
	Q_ASSERT(s.at(idx) == QLatin1Char('('));
	Q_ASSERT(s.at(l - 1) == QLatin1Char(')'));

	QStringList result;
	QString item;

	for (++idx; idx < l; ++idx) {
		QChar c = s.at(idx);
		if (c == QLatin1Char(')')) {
			Q_ASSERT(idx == l - 1);
			result.append(item);
		} else if (c == QLatin1Char(' ')) {
			result.append(item);
			item.clear();
		} else {
			item.append(c);
		}
	}

	return result;
}

QDataStream &operator<<(QDataStream &out, const PVCore::PVArgumentTypeBase &obj)
{
	obj.serialize(out);
	return out;
}

QDataStream &operator>>(QDataStream &in, const PVCore::PVArgumentTypeBase &obj)
{
	obj.unserialize(in);
	return in;
}

QString PVCore::PVArgument_to_QString(const PVCore::PVArgument &v)
{
	QString str;

	if (v.userType() >= QMetaType::User) { // custom type
		str = static_cast<PVArgumentTypeBase*>(const_cast<PVCore::PVArgument*>(&v)->data())->to_string();
	}
	else { // builtin type
		if (v.canConvert<QString>()) {
			str = v.toString();
		}
	}

	return str;
}

PVCore::PVArgument PVCore::QString_to_PVArgument(const QString &s, const QVariant& v, bool* res_ok /* = 0 */)
{
	QVariant var;
	bool ok = true;

	if (v.userType() >= QMetaType::User) { // custom type
		var = static_cast<const PVArgumentTypeBase*>(v.constData())->from_string(s, &ok);
	}
	else // builtin type
	{
		switch (v.type()) {
		case QMetaType::Bool:
			var = s.compare("true", Qt::CaseInsensitive) == 0;
			break;
		case QMetaType::Int:
			var = s.toInt(&ok);
			break;
		case QMetaType::UInt:
			var = s.toUInt(&ok);
			break;
		case QMetaType::Double:
			var = s.toDouble(&ok);
			break;
		case QMetaType::QChar:
			ok = s.length() >= 1;
			if (ok) {
				var = QVariant(QChar(s[0]));
			}
			break;
		case QMetaType::LongLong:
			var = s.toLongLong(&ok);
			break;
		case QMetaType::ULongLong:
			var = s.toULongLong(&ok);
			break;
		case QMetaType::QString:
			var = s;
			break;
		default:
			ok = false;
			break;
		}
	}

	if (!ok) {
		PVLOG_WARN("String '%s' can't be interpreted as a '%s' object ! Using default value...\n", qPrintable(s), v.typeName());
		var = v;
	}

	if (res_ok) {
		*res_ok = ok;
	}

	return var;
}

void PVCore::PVArgumentList_to_QSettings(const PVArgumentList& args, QSettings& settings, const QString& group_name)
{
	PVArgumentList::const_iterator it;
	settings.beginGroup(group_name);
	for (it = args.begin(); it != args.end(); it++) {
		settings.setValue(it.key(), PVArgument_to_QString(it.value()));
	}
	settings.endGroup();
}

PVCore::PVArgumentList PVCore::QSettings_to_PVArgumentList(QSettings& settings, const PVArgumentList& def_args, const QString& group_name)
{
	PVArgumentList args;
	settings.beginGroup(group_name);
	QStringList keys = settings.childKeys();
	for (int i = 0; i < keys.size(); i++) {
		QString const& key = keys.at(i);
		if (def_args.contains(key)) {
			QString str;
			if (settings.value(key).type() == QMetaType::QStringList) {
				// QSettings returns strings containing commas as QStringList
				str = settings.value(key).toStringList().join(",");
			}
			else {
				str = settings.value(key).toString();
			}
			args[key] = QString_to_PVArgument(str, def_args[key]);
		}
	}
	settings.endGroup();

	return args;
}

void PVCore::PVArgumentList_to_QDomElement(const PVArgumentList& args, QDomElement& elt)
{
	PVArgumentList::const_iterator it;
	for (it = args.begin(); it != args.end(); it++) {
		QDomElement arg_elt = elt.ownerDocument().createElement("argument");
		arg_elt.setAttribute("name", it.key());
		arg_elt.setAttribute("value", PVArgument_to_QString(it.value())); 
		elt.appendChild(arg_elt);
	}
}

PVCore::PVArgumentList PVCore::QDomElement_to_PVArgumentList(QDomElement const& elt, const PVArgumentList& def_args)
{
	// TODO: refaire ça !
	PVArgumentList args;
	QDomElement child = elt.firstChildElement("argument");
	for (; !child.isNull(); child = child.nextSiblingElement("argument")) {
		QString key = child.attribute("name", "");
		if (!def_args.contains(key)) {
			continue;
		}
		QString value = child.attribute("value", QString());
		args[key] = QString_to_PVArgument(value, def_args[key]);
	}
	return args;
}

void PVCore::dump_argument_list(PVArgumentList const& l)
{
	PVCore::PVArgumentList::const_iterator it;
	for (it = l.begin(); it != l.end(); it++) {
		PVLOG_INFO("%s = %s (%s)\n", qPrintable(it.key().key()), qPrintable(it.value().toString()), qPrintable(PVArgument_to_QString(it.value())));
	}
}

PVCore::PVArgumentList PVCore::filter_argument_list_with_keys(PVArgumentList const& args, PVArgumentKeyList const& keys, PVArgumentList const& def_args)
{
	PVCore::PVArgumentList ret;
	foreach (QString const& key, keys) {
		if (!def_args.contains(key)) {
			continue;
		}
		PVCore::PVArgument arg;
		if (args.contains(key)) {
			arg = args[key];
		}
		else {
			arg = def_args[key];
		}
		ret[key] = arg;
	}
	return ret;
}

void PVCore::PVArgumentList_set_common_args_from(PVCore::PVArgumentList& ret, PVCore::PVArgumentList const& ref)
{
	PVCore::PVArgumentList::iterator it;
	for (it = ret.begin(); it != ret.end(); it++) {
		QString const& key(it.key());
		if (ref.contains(key)) {
			it.value() = ref[key];
		}
	}
}

unsigned int qHash(PVCore::PVArgumentKey const& key)
{
	return qHash(key.key());
}
