//! \file PVCore::PVFieldSplitterURL.cpp
//! $Id: PVFieldSplitterURL.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2012
//! Copyright (C) Philippe Saadé 2011-2012
//! Copyright (C) Picviz Labs 2011-2012

// Handles urls like this:
// lintranet.beijaflore.com:443
// clients1.google.com:443

#include "PVFieldSplitterURL.h"
#include <pvkernel/core/PVBufferSlice.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/rush/PVAxisTagsDec.h>

#include <QUrl>

#define URL_NUMBER_FIELDS_CREATED 8

struct url_decode_buf {
	PVCore::PVField *pf[URL_NUMBER_FIELDS_CREATED];
	PVCore::PVField* field;
	char* data;
	size_t rem_len;
	size_t nelts;
};

static void url_decode_add_field(url_decode_buf* buf, QString const& new_field, PVCol pos)
{
	if (pos == -1) {
		return;
	}
	size_t bufsize = new_field.size() * sizeof(QChar);
	PVCore::PVField *pf = buf->pf[pos];
	if (bufsize > buf->rem_len) {
		// This buffer is too short... Do not reallocate it, since it will invalidate
		// previous pointers !!
		// Allocate a new one for 'pf'.
		pf->allocate_new(bufsize);
		memcpy(pf->begin(), new_field.constData(), bufsize);
	}
	else {
		memcpy(buf->data, new_field.constData(), bufsize);
		pf->set_begin(buf->data);
		pf->set_end(buf->data+bufsize);
		pf->set_physical_end(buf->data+bufsize);
		//PVCore::PVField f(*buf->parent, buf->data, buf->data+bufsize);

		buf->data += bufsize;
		buf->rem_len -= bufsize;
		buf->nelts++;
	}
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
		PVLOG_WARN("(PVFieldSplitterURL) unknown port for %s.\n", qPrintable(str));
		return false;
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

	// Default tags position values (if the splitter is used outside a format)
	_col_proto = 0;
	_col_subdomain = 1;
	_col_host = 2;
	_col_domain = 3;
	_col_tld = 4;
	_col_port = 5;
	_col_url = 6;
	_col_variable = 7;
	_ncols = 8;
}

void PVFilter::PVFieldSplitterURL::set_children_axes_tag(filter_child_axes_tag_t const& axes)
{
	PVFieldsBaseFilter::set_children_axes_tag(axes);
	_col_proto = axes.value(PVAXIS_TAG_PROTOCOL, -1);
	_col_subdomain = axes.value(PVAXIS_TAG_SUBDOMAIN, -1);
	_col_host = axes.value(PVAXIS_TAG_HOST, -1);
	_col_domain = axes.value(PVAXIS_TAG_DOMAIN, -1);
	_col_tld = axes.value(PVAXIS_TAG_TLD, -1);
	_col_port = axes.value(PVAXIS_TAG_PORT, -1);
	_col_url = axes.value(PVAXIS_TAG_URL, -1);
	_col_variable = axes.value(PVAXIS_TAG_URL_VARIABLES, -1);
	PVCol nmiss = (_col_proto == -1) + (_col_subdomain == -1) + (_col_host == -1) + (_col_domain == -1) + (_col_tld == -1) + (_col_port == -1) + \
				(_col_url == -1) + (_col_variable == -1);
	_ncols = URL_NUMBER_FIELDS_CREATED-nmiss;
	if (_ncols == 0) {
		PVLOG_WARN("(PVFieldSplitterURL::set_children_axes_tag) warning: URL splitter set but no tags have been found !\n");
	}
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterURL::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldSplitterURL::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	QString qstr;
	field.get_qstr(qstr);
	QString qstr_copy(qstr.constData(), qstr.size());
	QString none;		// usefull variable to put an empty string in fields

	// URL decoder buffer
	url_decode_buf buf;
	buf.data = field.begin();
	buf.rem_len = field.size();
	buf.field = &field;
	buf.nelts = 0;

	// Add the number of final fields and save their pointers
	PVCore::PVField ftmp(*field.elt_parent());
	for (PVCol i = 0; i < _ncols; i++) {
		PVCore::list_fields::iterator it_new = l.insert(it_ins, ftmp);
		buf.pf[i] = &(*it_new);
	}

	// URL splitter
	QUrl url(qstr, QUrl::TolerantMode);
	if (!url.isValid()) {
		QString ip;
		uint16_t port = 0;
		if (split_ip_port(qstr, ip, port)) {
			url_decode_add_field(&buf, none, _col_proto); // Protocol
			url_decode_add_field(&buf, none, _col_subdomain); // TLD
			url_decode_add_field(&buf, none, _col_host); // TLD
			url_decode_add_field(&buf, ip, _col_domain); // Domain
			url_decode_add_field(&buf, none, _col_tld); // TLD
			url_decode_add_field(&buf, QString::number(port), _col_port); // Port
			url_decode_add_field(&buf, none, _col_url); // URL
			url_decode_add_field(&buf, none, _col_variable); // Variable
			return buf.nelts;
		}
		PVLOG_WARN_FIELD(field, "(PVFieldSplitterURL) invalid url '%s'", qPrintable(qstr));
		field.set_invalid();
		field.elt_parent()->set_invalid();
		return 0;
	}

	QString port;
	QString host(url.host());
	QString url_path(url.path());
	QString qitems;

	int prepend_protocol = 0;

	// We test if we have :// in the begining of the url so we can add
	// the protocol properly instead of having bugs such as a procol named "foo.bar.com"
	// because the given url was "foo.bar.com:google.com/"
	int ret = qstr.indexOf(QString("://"));
	if ((ret > 1) && (ret < 15)) {
		url_decode_add_field(&buf, url.scheme(), _col_proto);
	} else {
		// We set prepend protocol to 1 since it is actually a host
		prepend_protocol = 1;
		url_decode_add_field(&buf, none, _col_proto);
	}

	if (prepend_protocol) {
		if (!url.scheme().isEmpty()) {
			// We must check if host is empty or not: that means that QUrl did put the host into url.scheme()
			// So we just put the scheme into the host by prepending it.
			if (!host.isEmpty()) {
				// Why we prepend ':' here? because it was what we saw when we
				// had the bug putting a host in the protocol part, as it was splitted
				// with ':'
				host.prepend(QString(":"));
			}
			host.prepend(url.scheme());
		} 
	}

	if (host.isEmpty()) {
		// We cannot decode the host because we have a url like:
		// foobar.com/index.html
		// however, foobar.com is a host!
		QStringList slashhost = url_path.split("/");

		host = slashhost[0];
		url_path.remove(0, host.size());
	}
	url_decode_add_field(&buf, host, _col_domain);

	// Now we play with the host(domain) string to extract the host and subdomain
	QStringList splitted_domain = host.split(".");
	if (splitted_domain.size() > 2) {
		QString just_host = splitted_domain[splitted_domain.size() -2] + QString(".") + splitted_domain[splitted_domain.size() - 1];
		splitted_domain.removeLast();
		splitted_domain.removeLast();
		QString just_subdomain = splitted_domain.join(".");
		url_decode_add_field(&buf, just_subdomain, _col_subdomain);		
		url_decode_add_field(&buf, just_host, _col_host);
	} else {
		url_decode_add_field(&buf, none, _col_subdomain);		
		url_decode_add_field(&buf, none, _col_host);
	}

	// TLD
	if (host.size() > 1 && host.at(host.size()-1).isDigit()) {
		// It should be an IP, at least a TLD ending with a number is weird
		url_decode_add_field(&buf, "", _col_tld);
	} else {
		QStringList dotlist = host.split(".");
		if (dotlist.size() > 1) {
			url_decode_add_field(&buf, dotlist[dotlist.size()-1], _col_tld);
		} else {
			// We have something like http://localhost/
			url_decode_add_field(&buf, "", _col_tld);
		}
	}

	// Port discovery:
	// Add port with 80 as default is there is no known port
	if (url.port() < 0) {
		port.prepend("80");
	} else {
		port.prepend(QString::number(url.port()));
	}

	if (!url_path.startsWith("/")) {
		// our url path does not starts with a '/'. The QUrl messed up with
		// the source port

		// it can be 443/foo.html
		if (url_path.contains("/")) {
			QStringList slashhost = url_path.split("/");
			port = slashhost[0];
			url_path.remove(0, slashhost[0].length());
		} else {
			port = url_path;
			url_path.remove(0, port.length());
		}
	} 
	// url_decode_add_field(&buf, port.number(url.port(80)));
	url_decode_add_field(&buf, port, _col_port);

	// PVLOG_INFO("url path (%s)\n", qPrintable(url_path));
	// int tryport = url.port(0);
	// if (tryport == 0) {
	// 	int tryport = url.port(44952);
	// 	// We cannot guess the source port
	// 	// Sometime the port information is put in the url_path. 
	// 	// Since a url must have a / as the first path char, we can
	// 	// easily check that (and check we have a number ;-)
	// 	if (url_path.contains("/")) {
	// 		
	// 	}
	// }

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
	url_decode_add_field(&buf, url_path, _col_url);
	url_decode_add_field(&buf, qstr_copy, _col_variable);

	return buf.nelts;
}


IMPL_FILTER_NOPARAM(PVFilter::PVFieldSplitterURL)
