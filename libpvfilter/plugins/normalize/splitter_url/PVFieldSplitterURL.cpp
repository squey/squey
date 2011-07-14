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

	QString none;		// usefull variable to put an empty string in fields

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
	QUrl url(field.qstr(), QUrl::TolerantMode);
	if (!url.isValid()) {
		QString ip;
		uint16_t port = 0;
		if (split_ip_port(field.qstr(), ip, port)) {
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

	QString value;
	QString host(url.host());
	QString url_path(url.path());
	QString qitems;

	int prepend_protocol = 0;

	// We test if we have :// in the begining of the url so we can add
	// the protocol properly instead of having bugs such as a procol named "foo.bar.com"
	// because the given url was "foo.bar.com:google.com/"
	int ret = field.qstr().indexOf(QString("://"));
	if ((ret > 1) && (ret < 15)) {
		url_decode_add_field(&buf, url.scheme());
	} else {
		// We set prepend protocol to 1 since it is actually a host
		prepend_protocol = 1;
		url_decode_add_field(&buf, none);
	}

	if (host.isEmpty()) {
		// We cannot decode the host because we have a url like:
		// foobar.com/index.html
		// however, foobar.com is a host!
		QStringList slashhost = url_path.split("/");

		host = slashhost[0];
		url_path.remove(0, host.size());
	}
	if ((prepend_protocol) && (!url.scheme().isEmpty())) {
		// Why we prepend ':' here? because it was what we saw when we
		// had the bug putting a host in the protocol part, as it was splitted
		// with ':'
		host.prepend(QString(":"));
		host.prepend(url.scheme());
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
