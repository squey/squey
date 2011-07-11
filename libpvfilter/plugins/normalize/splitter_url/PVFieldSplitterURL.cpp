//! \file PVCore::PVFieldSplitterURL.cpp
//! $Id: PVFieldSplitterURL.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include "PVFieldSplitterURL.h"
#include <pvcore/PVBufferSlice.h>

#include <QUrl>

struct url_decode_buf {
	PVCore::list_fields *l;
	PVCore::list_fields::iterator it_ins;
	PVCore::PVElement* parent;
	PVCore::PVField* field;
	char* data;
	size_t rem_len;
	size_t nelts;
};

static void url_decode_add_field(url_decode_buf* buf, QString const& new_field)
{
	size_t bufsize = new_field.size() * sizeof(QChar);
	if (bufsize > buf->rem_len) {
		size_t cur_index = buf->data - buf->field->begin();
		size_t growby = bufsize - buf->rem_len;
		buf->field->grow_by_reallocate(growby);
		buf->rem_len += growby;
		// Update the current data pointer
		buf->data = buf->field->begin() + cur_index;
	}
	memcpy(buf->data, new_field.constData(), bufsize);
	// Make a copy of the old field, in case a reallocation has been done (so that our field hold a ref
	// on the shared buffer, and that one won't be deleted on the original field delete)
	PVCore::PVField f(*buf->field);
	f.set_begin(buf->data);
	f.set_end(buf->data+bufsize);
	f.set_physical_end(f.end());
	f.init_qstr();

	buf->data += bufsize;
	buf->rem_len -= bufsize;
	buf->l->insert(buf->it_ins, f);
	buf->nelts++;
}

bool split_ip_port(QString const& str, QString& ip, uint16_t& port)
{
	QStringList l = str.trimmed().split(QChar(':'), QString::KeepEmptyParts);
	if (l.size() > 2) {
		return false;
	}

	if (l.size() == 2) {
		bool conv_ok = false;
		port = l[1].toUShort(&conv_ok);
		if (!conv_ok) {
			return false;
		}
	}
	else {
		PVLOG_WARN("(splitter_url) unknown port for %s. Port is set to 0.\n", qPrintable(str));
		port = 0;
	}

	ip = l[0];
	return true;
}

/******************************************************************************
 *
 * PVFilter::PVCore::PVFieldSplitterURL::PVCore::PVFieldSplitterURL
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterURL::PVFieldSplitterURL() :
	PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER_NOPARAM(PVFilter::PVFieldSplitterURL);
}


/******************************************************************************
 *
 * PVFilter::PVFieldSplitterURL::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldSplitterURL::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	field.init_qstr();

	// URL decoder buffer
	url_decode_buf buf;
	buf.l = &l;
	buf.it_ins = it_ins;
	buf.parent = field.elt_parent();
	buf.data = field.begin();
	buf.rem_len = field.size();
	buf.field = &field;
	buf.nelts = 0;

	// URL splitter
	QUrl url(field.qstr(), QUrl::StrictMode);
	if (!url.isValid()) {
		url.setUrl(field.qstr(), QUrl::TolerantMode);
		if (!url.isValid()) {
			QString ip;
			uint16_t port = 0;
			if (split_ip_port(field.qstr(), ip, port)) {
				QString none;
				url_decode_add_field(&buf, none); // Protocol
				url_decode_add_field(&buf, ip); // Domain
				url_decode_add_field(&buf, none); // TLD
				url_decode_add_field(&buf, QString::number(port)); // Port
				url_decode_add_field(&buf, none); // URL
				url_decode_add_field(&buf, none); // Variable
				return buf.nelts;
			}
			PVLOG_WARN("(PVFieldSplitterURL) invalid url '%s', cannot normalize\n", qPrintable(field.qstr()));
			field.set_invalid();
			field.elt_parent()->set_invalid();
			return 0;
		}
	}

	QString value;
	QString host(url.host());
	QString url_path(url.path());
	QString qitems;

	url_decode_add_field(&buf, url.scheme());
	if (host.isEmpty()) {
		// We cannot decode the host because we have a url like:
		// foobar.com/index.html
		// however, foobar.com is a host!
		QStringList slashhost = url_path.split("/");

		host = slashhost[0];
		url_path.remove(0, host.size());
	}
	url_decode_add_field(&buf, host);

	// TLD
	if (host.at(host.size()-1).isDigit()) {
		// It should be an IP, at least a TLD ending with a number is weird
		url_decode_add_field(&buf, "");
	} else {
		QStringList dotlist = host.split(".");
		if (dotlist.size() > 1) {
			url_decode_add_field(&buf, dotlist[dotlist.size()-1]);
		} else {
			// We have something like http://localhost/
			url_decode_add_field(&buf, "");
		}
	}
	url_decode_add_field(&buf, value.number(url.port(80)));

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
	url_decode_add_field(&buf, url_path);
	url_decode_add_field(&buf, qitems);

	return buf.nelts;
}

IMPL_FILTER_NOPARAM(PVFilter::PVFieldSplitterURL)
