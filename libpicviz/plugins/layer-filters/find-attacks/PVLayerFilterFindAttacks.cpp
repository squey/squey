//! \file PVLayerFilterFindAttacks.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include "PVLayerFilterFindAttacks.h"

#include <picviz/PVColor.h>
#include <pvcore/PVAxisIndexType.h>

#include <QDir>
#include <QString>

static QString snort_sigs_path;
static QString nessus_sigs_path;
static QHash<QString, QStringList> signatures;

/******************************************************************************
 *
 * Picviz::PVLayerFilterFindAttacks::PVLayerFilterFindAttacks
 *
 *****************************************************************************/
Picviz::PVLayerFilterFindAttacks::PVLayerFilterFindAttacks(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterFindAttacks, l);

	PVLOG_INFO("[%s] Initializing...\n", __FUNCTION__);
	snort_sigs_path = pvconfig.value("layer-filter-attack-finder/snort_sigs_path").toString();
	PVLOG_INFO("[%s] Snort plugins path: '%s'\n", __FUNCTION__, qPrintable(snort_sigs_path));
	nessus_sigs_path = pvconfig.value("layer-filter-attack-finder/nessus_sigs_path").toString();
	PVLOG_INFO("[%s] Nessus plugins path: '%s'\n", __FUNCTION__, qPrintable(nessus_sigs_path));


	PVLOG_INFO("[%s] Indexing Snort data...\n", __FUNCTION__);
	QDir snort_dir(snort_sigs_path);
	snort_dir.setFilter(QDir::Files | QDir::NoSymLinks);
	QStringList filters;
	filters << "*.rules";
	snort_dir.setNameFilters(filters);
	if (!snort_dir.exists()) {
		PVLOG_ERROR("[%s] Cannot find the snort directory: '%s'\n", __FUNCTION__, qPrintable(snort_sigs_path));
	} else {
		QFileInfoList files_list = snort_dir.entryInfoList();

		QRegExp snort_useragent_rx(".*User-Agent\\|3a\\| ([^\";\\|]*)\"; .*");

		for (int i = 0; i < files_list.size(); ++i) {
			QFileInfo fileInfo = files_list.at(i);
			QString rule_path = snort_sigs_path + PICVIZ_PATH_SEPARATOR + fileInfo.fileName();
			PVLOG_INFO("[%s] indexing %s\n", __FUNCTION__, qPrintable(rule_path));
			QFile snort_rule(rule_path);
			QByteArray data;

			if (!snort_rule.open(QIODevice::ReadOnly | QIODevice::Text)) {
				PVLOG_INFO("Cannot open %s\n", qPrintable(rule_path));
				continue;
			}
			
			data = snort_rule.readLine();
			while (!data.isEmpty()) {
				data.chop(1);
				if (data.isEmpty()) {
					data = snort_rule.readLine();
					continue;
				}

				if (data.at(0) == '#') {
					// We have a comment, we ignore
					continue;
				}
				// PVLOG_INFO("data=%s\n", data.data());
				int unused_pos = snort_useragent_rx.indexIn(data);
				QStringList rxlist = snort_useragent_rx.capturedTexts();
				rxlist.removeFirst();
				if (snort_useragent_rx.exactMatch(data)) {
					if (!rxlist[0].isEmpty()) {
						if (rxlist[0].length() > 3) { // 3 is prone to false positives
							signatures["Snort User Agents"] << rxlist[0];
						}
						// PVLOG_INFO("MATCH '%s'\n", qPrintable(rxlist[0]));
					}
				}

				data = snort_rule.readLine();
			}
			snort_rule.close();
		}
	}

	// QDir nessus_dir(nessus_sigs_path);
	// if (!nessus_dir.exists()) {
	// 	PVLOG_ERROR("Cannot find the nessus directory: '%s'\n", qPrintable(nessus_sigs_path));
	// } else {
	// 	// Nessus code
	// }

}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterFindAttacks)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterFindAttacks)
{
	PVCore::PVArgumentList args;

	args["Domain Axis"].setValue(PVCore::PVAxisIndexType(0));
	args["URL Axis"].setValue(PVCore::PVAxisIndexType(0));
	args["Variable Axis"].setValue(PVCore::PVAxisIndexType(0));
	args["User Agent Axis"].setValue(PVCore::PVAxisIndexType(0));

	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterFindAttacks::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterFindAttacks::operator()(PVLayer& /*in*/, PVLayer &out)
{	
	int ua_axis_id = _args["User Agent Axis"].value<PVCore::PVAxisIndexType>().get_original_index();
	PVRow nb_lines = _view->get_qtnraw_parent().size();

	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();

	QStringList snort_ua = signatures["Snort User Agents"];
	for (PVRow r = 0; r < nb_lines; r++) {
		PVRush::PVNraw::nraw_table_line const& nraw_r = nraw.at(r);

		// We search for user agents
		for (int i=0; i < snort_ua.size(); i++) {
			if (snort_ua[i].length() == nraw_r[ua_axis_id].length()) {
				QStringMatcher matcher(snort_ua[i]);
				int retval = matcher.indexIn(nraw_r[ua_axis_id]);
				if (retval >= 0) {
					PVLOG_INFO("find string '%s' in '%s'\n", qPrintable(snort_ua[i]), qPrintable(nraw_r[ua_axis_id]));
				}
				PVLOG_INFO("we set the line %d to %d\n", r, retval < 0 ? 0 : 1);
				out.get_selection().set_line(r, retval < 0 ? 0 : 1);			
			}
		}

	}

}

IMPL_FILTER(Picviz::PVLayerFilterFindAttacks)
