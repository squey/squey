#include <stdlib.h>

#include <QHash>
#include <QString>
#include <QVector>

#include <picviz/PVMapping.h>

LibCPPExport void picviz_mandatory_mapping_exec(Picviz::PVMapping_p mapping, pvrow row, pvcol col, QString &value, float mapped_pos, void *userdata, bool is_first)
{
	float ymin;
	float ymax;

	QHash<QString, float> float_hash;
	QHash<QString, QString> string_hash;

/*	if (is_first) {
		float_hash["ymin"] = mapped_pos;
		float_hash["ymax"] = mapped_pos;

		string_hash["ymin"] = value;
		string_hash["ymax"] = value;

		mapping->dict_float.append(float_hash);
		mapping->dict_string.append(string_hash);
	} else {
		float_hash = mapping->dict_float[col];
		string_hash = mapping->dict_string[col];

		ymin = float_hash["ymin"];
		ymax = float_hash["ymax"];

		if (mapped_pos < ymin) {
			mapping->dict_float[col]["ymin"] = mapped_pos;
			mapping->dict_string[col]["ymin"] = value;
		}
		if (mapped_pos > ymax) {
			mapping->dict_float[col]["ymax"] = mapped_pos;
			mapping->dict_string[col]["ymax"] = value;
		}
	}*/
}

LibCPPExport int picviz_mandatory_mapping_init()
{
	return 0;
}

LibCPPExport int picviz_mandatory_mapping_terminate()
{
	return 0;
}

LibCPPExport int picviz_mandatory_mapping_test()
{
	return 0;
}


