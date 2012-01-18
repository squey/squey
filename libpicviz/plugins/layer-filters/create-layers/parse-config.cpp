#include <QByteArray>
#include <QRegExp>
#include <QFile>

#include <pvkernel/core/general.h>

#include "parse-config.h"

int create_layers_parse_config(QString filename, int (*handle_create_layers_section)(QString section_name, QMap<QString, QStringList> layers_regex))
{
	// PVLOG_INFO("Parse config for %s\n", filename);
	QMap<QString, QMap<QString, QStringList> > section_layer_regex_list;

	QRegExp re_section("^\\[(.*)\\]$", Qt::CaseSensitive, QRegExp::RegExp);
	QRegExp re_layer("([a-zA-Z ]+)[^a-zA-Z ](.*)", Qt::CaseSensitive, QRegExp::RegExp);

	QString section_name;
	QString layer_name;
	QString regex_for_layer;
       	int line_pos = 1;

	int fp_ret = 0;

	if (!handle_create_layers_section) {
		PVLOG_ERROR("%s: NULL function pointer. Nothing to do!\n", __FUNCTION__);
		return 1;
	}

	QFile configfile(filename);
	if (!configfile.open(QIODevice::ReadOnly | QIODevice::Text)) {
		return -1;
	}

	while (!configfile.atEnd()) {
		QByteArray line = configfile.readLine();
		line.chop(1);
		if ( (!line.isEmpty()) || (!line[0] != '#') ) {
			int ret = re_section.indexIn(line);
			if (ret >= 0) {
			        section_name = re_section.cap(1);
				// PVLOG_INFO("Section:%s\n", qPrintable(section_name));
			} else {
				if (section_name.isEmpty()) {
					PVLOG_ERROR("Create Layer config issue: empty section found at line %d\n", line_pos);
				} else {
				// We know in which section we are and we parse data
					if (!section_name.isEmpty()) {
						int ret = re_layer.indexIn(line);
						if (ret >= 0) {
							layer_name = re_layer.cap(1);
							regex_for_layer = re_layer.cap(2);

							PVLOG_DEBUG("section '%s', key '%s', value '%s'\n", qPrintable(section_name), qPrintable(layer_name), qPrintable(regex_for_layer));
							if (handle_create_layers_section) {
								section_layer_regex_list[section_name][layer_name] << regex_for_layer;
								// fp_ret += handle_create_layers_section(section_name, layer_name, regex_for_layer);
							} else {
							}
						}
					}
					
				}
			}
		}

		line_pos++;
	}

	QMapIterator<QString, QMap<QString, QStringList> > i(section_layer_regex_list);
	while (i.hasNext()) {
		i.next();
		QMap<QString, QStringList> layers_regex;

		QMapIterator<QString, QStringList> j(i.value());
		while (j.hasNext()) {
			j.next();
			layers_regex[j.key()] = j.value();
		}
		fp_ret += handle_create_layers_section(i.key(), layers_regex);
	}

	return fp_ret;
}
