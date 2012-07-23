/**
 * \file PVVersion.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVVersion.h>
#include <pvkernel/core/stdint.h>


#include <QString>
#include <QStringList>

bool PVCore::PVVersion::from_network_reply(QByteArray const& reply, version_t& current, version_t& last)
{
	// Format is:
	// MAJOR\n
	// MINOR\n
	// PATCH\n
	// (for the current major/minor version)
	// MAJOR\n
	// MINOR\n
	// PATCH\n
	// (for the latest version)
	QString str = QString::fromUtf8(reply.data(), reply.size());
	QStringList versions = str.split('\n');
	if (versions.size() < 6) {
		// Well, just forget about it.
		PVLOG_DEBUG("(PVCore::PVVersion::from_network_reply) not enough fields returned (%d < 6).\n", versions.size());
		return false;
	}
	bool ok = false;
	uint8_t major = (uint8_t) versions[0].toUShort(&ok);
	if (!ok) {
		return false;
	}
	uint8_t minor = (uint8_t) versions[1].toUShort(&ok);
	if (!ok) {
		return false;
	}

	uint8_t patch = (uint8_t) versions[2].toUShort(&ok);
	if (!ok) {
		return false;
	}
	current = PICVIZ_VERSION(major, minor, patch);

	major = (uint8_t) versions[3].toUShort(&ok);
	if (!ok) {
		return false;
	}
	minor = (uint8_t) versions[4].toUShort(&ok);
	if (!ok) {
		return false;
	}

	patch = (uint8_t) versions[5].toUShort(&ok);
	if (!ok) {
		return false;
	}
	last = PICVIZ_VERSION(major, minor, patch);

	return true;
}

QString PVCore::PVVersion::to_str(version_t v)
{
	return QString("%1.%2.%3").arg(PICVIZ_MAJOR_VERSION(v)).arg(PICVIZ_MINOR_VERSION(v)).arg(PICVIZ_PATCH_VERSION(v));
}

QString PVCore::PVVersion::update_url()
{
	//return QString("http://files.geekou.info/pv-test/update-%1.%2").arg(PICVIZ_CURRENT_VERSION_MAJOR).arg(PICVIZ_CURRENT_VERSION_MINOR);
	return QString("http://www.picviz.com/update-%1.%2").arg(PICVIZ_CURRENT_VERSION_MAJOR).arg(PICVIZ_CURRENT_VERSION_MINOR);
}
