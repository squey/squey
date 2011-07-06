#include <stdlib.h>

#include <QStringList>
#include <QString>

#include <picviz/PVMapping.h>

LibCPPExport float picviz_mapping_exec(const Picviz::PVMapping_p mapping, PVCol index, QString &value, void *userdata, bool is_first)
{
        QString qvalue(value);
	float fval;

	QStringList values = qvalue.split("@");

	if (values.count() < 2) {
	        PVLOG_ERROR("%s: Cannot split the string in two objects. Missing '@' element to put position (ie. 'foo@0.5'). Returning 0!\n", __FUNCTION__);
	        return 0;
	}

	return values[1].toFloat();
}

LibCPPExport int picviz_mapping_init()
{
	return 0;
}

LibCPPExport int picviz_mapping_terminate()
{
	return 0;
}

LibCPPExport int picviz_mapping_test()
{
	return 0;
}


