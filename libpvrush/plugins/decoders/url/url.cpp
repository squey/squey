/*
 * $Id$
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 */

#include <pvcore/general.h>

#include <pvrush/PVFormat.h>

#include <QList>
#include <QStringList>
#include <QString>
#include <QHash>
#include <QUrl>

using namespace PVCore;

// We decode an url like http://www.example.com/foo.html?foo=bar&pic=viz
// into:
// http , www.example.com , com , 80 , /foo.html , ?foo=bar&pic=viz
int decode(int axis_id, QList<QStringList> *normalized_list)
{
	for (int i = 0; i < normalized_list->size(); ++i) {
  		QStringList *slist = (QStringList *) &normalized_list->at(i);
		QString str_url = slist->at(axis_id);
		QUrl url(str_url);
		QString value;
		QString host(url.host());
		QString url_path(url.path());
		QString qitems;
		
		if (!url.isValid()) {
			PVLOG_ERROR("Invalid url '%s'; Cannot normalize\n", str_url.toUtf8().data());
			continue;
		}

		slist->removeAt(axis_id);

		// We get all the query items and build a string from them
		QList<QPair<QByteArray, QByteArray> > query_items = url.encodedQueryItems();
		QListIterator<QPair<QByteArray, QByteArray> > query_items_i(query_items);
		while(query_items_i.hasNext()){
		  QPair<QByteArray, QByteArray> queryItem = query_items_i.next();
		  QByteArray normItem = queryItem.first + '=' + queryItem.second;
		  qitems.append(normItem);

		  if (query_items_i.hasNext()) {
		    qitems.append("&");
		  }
		}
		slist->insert(axis_id, qitems);
		

		if (host.isEmpty()) {
		  // We cannot decode the host because we have a url like:
		  // foobar.com/index.html
		  // however, foobar.com is a host!
		  QStringList slashhost = url_path.split("/");

		  host = slashhost[0];
		  url_path.remove(0, host.size());
		}
		slist->insert(axis_id, url_path);
		slist->insert(axis_id, value.number(url.port(80)));

		// TLD
		if (host.at(host.size()-1).isDigit()) {
		  // It should be an IP, at least a TLD ending with a number is weird
		  slist->insert(axis_id, "");
		} else {
		  QStringList dotlist = host.split(".");
		  if (dotlist.size() > 1) {
		    slist->insert(axis_id, dotlist[dotlist.size()-1]);
		  } else {
		    // We have something like http://localhost/
		    slist->insert(axis_id, "");
		  }
		}
		
		slist->insert(axis_id, host);
		slist->insert(axis_id, url.scheme());
  	}


	return 0;
}

// When we insert elements, we get the decoder axis id - 1 and do it last to first
LibCPPExport int pvrush_decoder_run(PVRush::PVFormat *format, QList<QStringList> *normalized_list, QHash<QString, QString> decoderopt)
{
	QHashIterator<int, QString> decoder_axis_hash(format->axis_decoder);
	int retval;

	while (decoder_axis_hash.hasNext()) {
		decoder_axis_hash.next();
		// log(loglevel::notice, "axis_decoder[%d]:'%s'\n", decoder_axis_hash.key(), decoder_axis_hash.value().toUtf8().data());
		if (!decoder_axis_hash.value().compare("url")) {
			retval += decode(decoder_axis_hash.key()-1, normalized_list);
		}
	}

	return 0;
}


