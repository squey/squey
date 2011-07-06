#include "PVMappingFilterUserdefinedDefault.h"
#include <QStringList>

float Picviz::PVMappingFilterUserdefinedDefault::operator()(QString const& value)
{
	QStringList p = value.split(QString::fromUtf8("@"));

	bool ok = false;
	float ret = p[1].toFloat(&ok);
	
	if (!ok) {
		PVLOG_WARN("(user-defined_default filter) unable to determine mapping from '%s'. Returns 0.\n", qPrintable(value));
		return 0;
	}

	return ret;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterUserdefinedDefault)
