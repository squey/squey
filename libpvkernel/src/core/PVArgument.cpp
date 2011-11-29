#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVTimeFormatType.h>
#include <QStringList>
#include <QRect>
#include <QMetaType>

QHash<QString,QString> PVCore::PVArgumentKey::_key_desc;

// Inspired from QSettingsPrivate functions !

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

QString PVCore::PVArgument_to_QString(const PVArgument &v)
{
	QString result;
	int vtype = v.userType();

	switch (vtype) {
		case QVariant::Invalid:
			result = QLatin1String("@Invalid()");
			break;

		case QVariant::ByteArray:
		{
			QByteArray a = v.toByteArray();
			result = QLatin1String("@ByteArray(");
			result += QString::fromLatin1(a.constData(), a.size());
			result += QLatin1Char(')');
			break;
		}

		// This is not supported in QSettings !!
		case QVariant::Char:
		{
			result = QLatin1String("@Char(");
			result += v.toString();
			result += QLatin1Char(')');
			break;
		}

		case QVariant::String:
		case QVariant::LongLong:
		case QVariant::ULongLong:
		case QVariant::Int:
		case QVariant::UInt:
		case QVariant::Bool:
		case QVariant::Double:
		case QVariant::KeySequence:
		{
			result = v.toString();
			if (result.startsWith(QLatin1Char('@')))
				result.prepend(QLatin1Char('@'));
			break;
		}
#ifndef QT_NO_GEOM_VARIANT
		case QVariant::Rect:
		{
			QRect r = qvariant_cast<QRect>(v);
			result += QLatin1String("@Rect(");
			result += QString::number(r.x());
			result += QLatin1Char(' ');
			result += QString::number(r.y());
			result += QLatin1Char(' ');
			result += QString::number(r.width());
			result += QLatin1Char(' ');
			result += QString::number(r.height());
			result += QLatin1Char(')');
			break;
		}
		case QVariant::Size:
		{
			QSize s = qvariant_cast<QSize>(v);
			result += QLatin1String("@Size(");
			result += QString::number(s.width());
			result += QLatin1Char(' ');
			result += QString::number(s.height());
			result += QLatin1Char(')');
			break;
		}
		case QVariant::Point:
		{
			QPoint p = qvariant_cast<QPoint>(v);
			result += QLatin1String("@Point(");
			result += QString::number(p.x());
			result += QLatin1Char(' ');
			result += QString::number(p.y());
			result += QLatin1Char(')');
			break;
		}
#endif // !QT_NO_GEOM_VARIANT

		default:
		{
			if (vtype == qMetaTypeId<PVCore::PVTimeFormatType>()) {
				result = QLatin1String("@PVTimeFormat(") + v.value<PVCore::PVTimeFormatType>().join("\n") + QLatin1String(")");
				break;
			}

#ifndef QT_NO_DATASTREAM
			QByteArray a;
			{
				QDataStream s(&a, QIODevice::WriteOnly);
				s.setVersion(QDataStream::Qt_4_0);
				s << v;
			}

			result = QLatin1String("@Variant(");
			result += QString::fromLatin1(a.constData(), a.size());
			result += QLatin1Char(')');
#else
			Q_ASSERT(!"QSettings: Cannot save custom types without QDataStream support");
#endif
			break;
		}
	}

	return result;
}


PVCore::PVArgument PVCore::QString_to_PVArgument(const QString &s)
{
	if (s.startsWith(QLatin1Char('@'))) {
		if (s.endsWith(QLatin1Char(')'))) {
			if (s.startsWith(QLatin1String("@ByteArray("))) {
				return QVariant(s.toLatin1().mid(11, s.size() - 12));
			} else if (s.startsWith(QLatin1String("@Char("))) {
				return (s.size() <= 7) ? QVariant() : QVariant(QChar(s[6]));
			} else if (s.startsWith(QLatin1String("@Variant("))) {
#ifndef QT_NO_DATASTREAM
				QByteArray a(s.toLatin1().mid(9));
				QDataStream stream(&a, QIODevice::ReadOnly);
				stream.setVersion(QDataStream::Qt_4_0);
				QVariant result;
				stream >> result;
				return result;
#else
				Q_ASSERT(!"QSettings: Cannot load custom types without QDataStream support");
#endif
#ifndef QT_NO_GEOM_VARIANT
			} else if (s.startsWith(QLatin1String("@Rect("))) {
				QStringList args = splitArgs(s, 5);
				if (args.size() == 4)
					return QVariant(QRect(args[0].toInt(), args[1].toInt(), args[2].toInt(), args[3].toInt()));
			} else if (s.startsWith(QLatin1String("@Size("))) {
				QStringList args = splitArgs(s, 5);
				if (args.size() == 2)
					return QVariant(QSize(args[0].toInt(), args[1].toInt()));
			} else if (s.startsWith(QLatin1String("@Point("))) {
				QStringList args = splitArgs(s, 6);
				if (args.size() == 2)
					return QVariant(QPoint(args[0].toInt(), args[1].toInt()));
#endif
			} else if (s.startsWith(QLatin1String("@PVTimeFormat("))) {
				QVariant ret;
				if (s.size() >= 14) {
					ret.setValue<PVCore::PVTimeFormatType>(PVCore::PVTimeFormatType(s.mid(14, s.size()-15).split("\n")));
				}
				return ret;
			} else if (s == QLatin1String("@Invalid()")) {
				return QVariant();
			}

		}
		if (s.startsWith(QLatin1String("@@")))
			return QVariant(s.mid(1));
	}

	return QVariant(s);
}

void PVCore::dump_argument_list(PVArgumentList const& l)
{
	PVCore::PVArgumentList::const_iterator it;
	for (it = l.begin(); it != l.end(); it++) {
		PVLOG_DEBUG("%s = %s (%s)\n", qPrintable(it.key().key()), qPrintable(it.value().toString()), qPrintable(PVArgument_to_QString(it.value())));
	}
}

unsigned int qHash(PVCore::PVArgumentKey const& key)
{
	return qHash(key.key());
}
