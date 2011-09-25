//! \file PVLayerFilterFindAttacks.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include "PVLayerFilterFindAttacks.h"

#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxisIndexType.h>
#include <picviz/PVView.h>

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


		// content:"/CFIDE/administrator/archives/index.cfm"; nocase; http_uri;
		QRegExp snort_httpuri_rx(".*content:\"([^;]*)\"; nocase; http_uri;.*");
		QRegExp snort_useragent_rx(".*User-Agent\\|3a\\| ([^\";\\|]*)\"; .*");

		for (int i = 0; i < files_list.size(); ++i) {
			QFileInfo fileInfo = files_list.at(i);
			QString rule_path = snort_sigs_path + PICVIZ_PATH_SEPARATOR + fileInfo.fileName();
			PVLOG_INFO("[%s] indexing %s\n", __FUNCTION__, qPrintable(rule_path));
			QFile snort_rule(rule_path);
			QByteArray data;
			int unused_pos;
			QStringList rxlist;

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
				unused_pos = snort_useragent_rx.indexIn(data);
				rxlist = snort_useragent_rx.capturedTexts();
				rxlist.removeFirst();
				if (snort_useragent_rx.exactMatch(data)) {
					if (!rxlist[0].isEmpty()) {
						signatures["Snort User Agents"] << rxlist[0];
						// PVLOG_INFO("MATCH '%s'\n", qPrintable(rxlist[0]));
					}
				}
				unused_pos = snort_httpuri_rx.indexIn(data);
				rxlist = snort_httpuri_rx.capturedTexts();
				rxlist.removeFirst();
				if (snort_httpuri_rx.exactMatch(data)) {
					if (!rxlist[0].isEmpty()) {
						signatures["Snort URI"] << rxlist[0];
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
void Picviz::PVLayerFilterFindAttacks::operator()(PVLayer& in, PVLayer &out)
{	
	int ua_axis_id = _args["User Agent Axis"].value<PVCore::PVAxisIndexType>().get_original_index();
	int url_axis_id = _args["URL Axis"].value<PVCore::PVAxisIndexType>().get_original_index();
	int variable_axis_id = _args["Variable Axis"].value<PVCore::PVAxisIndexType>().get_original_index();

	int retval;

	// bool ua_checked = _args["User Agent Axis"].value<PVCore::PVAxisIndexType>().get_checked();
	PVRow nb_lines = _view->get_qtnraw_parent().size();

	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();

	// PVLOG_INFO("UA AXIS ID=%d CHECKED=%d\n", ua_axis_id, ua_checked);

	out.get_selection().select_none();

	QStringList snort_ua = signatures["Snort User Agents"];
	QStringList snort_httpuri = signatures["Snort URI"];
	for (PVRow r = 0; r < nb_lines; r++) {
		if (should_cancel()) {
			if (&in != &out) {
				out = in;
			}
			return;
		}

		PVRush::PVNraw::nraw_table_line const& nraw_r = nraw.at(r);
		
		// We search for user agents
		// for (int i=0; i < snort_ua.size(); i++) {
		// 	if (snort_ua[i].length() == nraw_r[ua_axis_id].length()) {
		// 		// PVLOG_INFO("find string '%s' in '%s'\n", qPrintable(snort_ua[i]), qPrintable(nraw_r[ua_axis_id]));
		// 		QStringMatcher matcher(snort_ua[i]);
		// 		int retval = matcher.indexIn(nraw_r[ua_axis_id]);
		// 		if (retval >= 0) {
		// 			out.get_selection().set_line(r, 1);
		// 			break;
		// 		}
		// 	}
		// }

		// We search for URI
		for (int i=0; i < snort_httpuri.size(); i++) {
			if (should_cancel()) {
				if (&in != &out) {
					out = in;
				}
				return;
			}

			if (snort_httpuri[i].length() > 10) {
				QStringMatcher matcher_uri(snort_httpuri[i]);
				retval = matcher_uri.indexIn(nraw_r[url_axis_id]);
				if (retval >= 0) {
					out.get_selection().set_line(r, 1);
					break;
				}
			}

			// QStringMatcher matcher_variable(snort_httpuri[i]);
			// retval = matcher_variable.indexIn(nraw_r[variable_axis_id]);
			// if (retval >= 0) {
			// 	out.get_selection().set_line(r, 1);
			// 	break;
			// }
		}

	}

}

IMPL_FILTER(Picviz::PVLayerFilterFindAttacks)
