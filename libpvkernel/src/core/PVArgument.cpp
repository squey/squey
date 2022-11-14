//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVArgument.h> // for PVArgumentList, PVArgument, etc
#include <pvkernel/core/PVLogger.h>   // for PVLOG_INFO, PVLOG_WARN
#include <pvkernel/core/PVOrderedMap.h>

#include <QChar>
#include <QDomElement>
#include <QHash>
#include <QMetaType>
#include <QSettings>
#include <QString>
#include <QStringList>
#include <QVariant>
#include <QDataStream>

#include <vector> // for vector

QHash<QString, QString> PVCore::PVArgumentKey::_key_desc;

QDataStream& PVCore::operator<<(QDataStream& out, const PVCore::PVArgumentTypeBase& obj)
{
	obj.serialize(out);
	return out;
}

QDataStream& PVCore::operator>>(QDataStream& in, const PVCore::PVArgumentTypeBase& obj)
{
	obj.unserialize(in);
	return in;
}

QString PVCore::PVArgument_to_QString(const PVCore::PVArgument& v)
{
	QString str;

	if (v.userType() >= QMetaType::User) { // custom type
		str = static_cast<PVArgumentTypeBase*>(const_cast<PVCore::PVArgument*>(&v)->data())
		          ->to_string();
	} else { // builtin type
		if (v.canConvert<QString>()) {
			str = v.toString();
		} else if (v.canConvert<QStringList>()) {
			str = v.toStringList().join(",");
		}
	}

	return str;
}

PVCore::PVArgument
PVCore::QString_to_PVArgument(const QString& s, const QVariant& v, bool* res_ok /* = 0 */)
{
	QVariant var;
	bool ok = true;

	if (v.userType() >= QMetaType::User) { // custom type
		var = static_cast<const PVArgumentTypeBase*>(v.constData())->from_string(s, &ok);
	} else // builtin type
	{
		switch (v.typeId()) {
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
		case QMetaType::QStringList:
			var = s.split(",");
			break;
		default:
			ok = false;
			break;
		}
	}

	if (!ok) {
		PVLOG_WARN("String '%s' can't be interpreted as a '%s' object ! Using default value...\n",
		           qPrintable(s), v.typeName());
		var = v;
	}

	if (res_ok != nullptr) {
		*res_ok = ok;
	}

	return var;
}

void PVCore::PVArgumentList_to_QSettings(const PVArgumentList& args,
                                         QSettings& settings,
                                         const QString& group_name)
{
	settings.beginGroup(group_name);
	for (const auto& arg : args) {
		settings.setValue(arg.key(), PVArgument_to_QString(arg.value()));
	}
	settings.endGroup();
}

PVCore::PVArgumentList PVCore::QSettings_to_PVArgumentList(QSettings& settings,
                                                           const PVArgumentList& def_args,
                                                           const QString& group_name)
{
	PVArgumentList args;
	settings.beginGroup(group_name);
	QStringList keys = settings.childKeys();
	for (int i = 0; i < keys.size(); i++) {
		QString const& key = keys.at(i);
		if (def_args.contains(key)) {
			QString str;
			if (settings.value(key).typeId() == static_cast<QMetaType::Type>(QMetaType::QStringList)) {
				// QSettings returns strings containing commas as QStringList
				str = settings.value(key).toStringList().join(",");
			} else {
				str = settings.value(key).toString();
			}
			args[key] = QString_to_PVArgument(str, def_args.at(key));
		}
	}
	settings.endGroup();

	PVArgumentList_set_missing_args(args, def_args);

	return args;
}

void PVCore::PVArgumentList_to_QDomElement(const PVArgumentList& args, QDomElement& elt)
{
	for (const auto& arg : args) {
		QDomElement arg_elt = elt.ownerDocument().createElement("argument");
		arg_elt.setAttribute("name", arg.key());
		arg_elt.setAttribute("value", PVArgument_to_QString(arg.value()));
		elt.appendChild(arg_elt);
	}
}

PVCore::PVArgumentList PVCore::QDomElement_to_PVArgumentList(QDomElement const& elt,
                                                             const PVArgumentList& def_args)
{
	PVArgumentList args;
	QDomElement child = elt.firstChildElement("argument");
	for (; !child.isNull(); child = child.nextSiblingElement("argument")) {
		QString key = child.attribute("name", "");
		if (!def_args.contains(key)) {
			continue;
		}
		QString value = child.attribute("value", QString());
		args[key] = QString_to_PVArgument(value, def_args.at(key));
	}
	return args;
}

void PVCore::dump_argument_list(PVArgumentList const& l)
{
	for (auto it = l.begin(); it != l.end(); it++) {
		PVLOG_INFO("%s = %s (%s)\n", qPrintable(it->key().key()),
		           qPrintable(it->value().toString()),
		           qPrintable(PVArgument_to_QString(it->value())));
	}
}

PVCore::PVArgumentList PVCore::filter_argument_list_with_keys(PVArgumentList const& args,
                                                              PVArgumentKeyList const& keys,
                                                              PVArgumentList const& def_args)
{
	PVCore::PVArgumentList ret;
	for (QString const& key : keys) {
		if (!def_args.contains(key)) {
			continue;
		}
		PVCore::PVArgument arg;
		if (args.contains(key)) {
			arg = args.at(key);
		} else {
			arg = def_args.at(key);
		}
		ret[key] = arg;
	}
	return ret;
}

void PVCore::PVArgumentList_set_common_args_from(PVCore::PVArgumentList& ret,
                                                 PVCore::PVArgumentList const& ref)
{
	for (auto& it : ret) {
		QString const& key(it.key());
		if (ref.contains(key)) {
			it.value() = ref.at(key);
		}
	}
}

void PVCore::PVArgumentList_set_missing_args(PVCore::PVArgumentList& ret,
                                             PVCore::PVArgumentList const& def_args)
{
	for (const auto& def_arg : def_args) {
		QString const& key(def_arg.key());
		if (!ret.contains(key)) {
			ret[key] = def_arg.value();
		}
	}
}

unsigned int qHash(PVCore::PVArgumentKey const& key)
{
	return qHash(key.key());
}
