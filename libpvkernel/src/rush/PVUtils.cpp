/**
 * \file PVUtils.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <QCryptographicHash>
#include <QFile>
#include <QString>

#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvkernel/rush/PVUtils.h>

/*
QString PVRush::PVUtils::generate_key_from_axes_values(PVCore::PVAxesIndexType const& axes, PVRush::PVNraw::const_nraw_table_line const& values)
{
	QString ret;
	PVCore::PVAxesIndexType::const_iterator it;
	for (it = axes.begin(); it != axes.end(); it++) {
		ret.append(values[*it].get_qstr());
	}
	return ret;
}
*/

const QByteArray PVRush::PVUtils::get_file_checksum(const QString& path)
{
	QFile file;
	file.setFileName(path);
	if (!file.open(QIODevice::ReadOnly)) {
		return QByteArray();
	}
	QByteArray data = file.readAll();
	file.close();

	return QCryptographicHash::hash(data, QCryptographicHash::Md5).toHex();
}

bool PVRush::PVUtils::files_have_same_content(const QString& path1, const QString& path2)
{
	return get_file_checksum(path1) == get_file_checksum(path2);
}

bool PVRush::PVUtils::safe_export(QString& str, const QString& sep_char, const QString& quote_char)
{
	static QString escaped_quote("\\" + quote_char);

	bool do_quote = false;

	if (str.contains(sep_char)) {
		do_quote = true;
	}
	if (str.contains(quote_char)) {
		do_quote = true;
		str.replace(quote_char, escaped_quote);
	}
	if (do_quote) {
		str.append(quote_char);
		str.prepend(quote_char);
	}

	return do_quote;
}

void PVRush::PVUtils::safe_export(QStringList& str_list, const QString& sep_char, const QString& quote_char)
{
	for (QString& str : str_list) {
		safe_export(str, sep_char, quote_char);
	}
}
