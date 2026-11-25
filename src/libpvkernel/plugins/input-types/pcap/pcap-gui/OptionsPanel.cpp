//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "OptionsPanel.h"

#include "Style.h"

#include <libpvpcap/ws.h>

#include <pvkernel/core/PVTheme.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <QCheckBox>
#include <QComboBox>
#include <QFileInfo>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QStandardPaths>
#include <QToolButton>

#include <algorithm>

namespace PVPcapsicum
{

OptionsPanel::OptionsPanel(QWidget* parent) : QWidget(parent)
{
	setAttribute(Qt::WA_StyledBackground);
	auto* grid = new QGridLayout(this);
	grid->setContentsMargins(12, 12, 12, 12);
	grid->setHorizontalSpacing(16);
	grid->setVerticalSpacing(3);
	grid->setColumnStretch(1, 1);
	int row = 0;
	const auto add = [&](const QString& name, QLayout* controls, const QString& description) {
		if (row != 0) {
			grid->setRowMinimumHeight(row++, 14);
		}
		grid->addWidget(new QLabel(name), row, 0);
		grid->addLayout(controls, row++, 1);
		auto* explanation = new QLabel(description);
		explanation->setWordWrap(true);
		style::color_label(explanation, style::secondary());
		grid->addWidget(explanation, row++, 1);
	};
	const auto line = [](std::initializer_list<QWidget*> widgets) {
		auto* layout = new QHBoxLayout;
		layout->setSpacing(16);
		for (QWidget* widget : widgets) {
			layout->addWidget(widget);
		}
		layout->addStretch();
		return layout;
	};

	_reassemble = new QCheckBox(tr("TCP streams"));
	_defragment = new QCheckBox(tr("IP fragments"));
	add(tr("Reassembly"), line({_reassemble, _defragment}),
	    tr("With reassembly, the fields of a message sent in several TCP segments or IP "
	       "fragments, such as a large HTTP response, come with its last frame. Without it, "
	       "each frame is decoded on its own."));

	_occurrence = new QComboBox;
	_occurrence->setObjectName("occurrence_combo");
	_occurrence->addItem(tr("keep their first value"), "f");
	_occurrence->addItem(tr("keep their last value"), "l");
	_occurrence->addItem(tr("keep all their values, as text"), "a");
	_aggregator = new QLineEdit;
	_aggregator->setMaxLength(1);
	_aggregator->setFixedWidth(_aggregator->fontMetrics().horizontalAdvance("MM") + 16);
	QHBoxLayout* occurrence = line({_occurrence, new QLabel(tr("separated by")), _aggregator});
	occurrence->setSpacing(8);
	add(tr("Repeated fields"), occurrence,
	    tr("What a column holds when its field occurs several times in a frame, such as the "
	       "addresses of a tunnel."));

	_wireshark_profile = new QComboBox;
	_wireshark_profile->setObjectName("wireshark_profile_combo");
	_wireshark_profile->addItem(tr("none"), QString());
	for (const std::string& path : pvpcap::get_wireshark_profiles_paths()) {
		const QString name = QFileInfo(QString::fromStdString(path)).fileName();
		_wireshark_profile->addItem(name, name);
	}
	_wireshark_profiles = _wireshark_profile->count();
	add(tr("Wireshark profile"), line({_wireshark_profile}),
	    tr("The preferences, \"Decode As\" rules and protocols turned off of a configuration "
	       "profile of Wireshark, such as one decoding a protocol on an unusual port."));
	_wireshark_problem = new QLabel;
	_wireshark_problem->setObjectName("wireshark_problem");
	_wireshark_problem->setWordWrap(true);
	style::color_label(_wireshark_problem, style::warning());
	grid->addWidget(_wireshark_problem, row++, 1);

	_resolve = new QCheckBox(tr("Resolve the addresses"));
	_from_captures = new QCheckBox(tr("from the DNS answers of the captures"));
	_from_servers = new QCheckBox(tr("asking DNS servers"));
	add(tr("Host names"), line({_resolve, _from_captures, _from_servers}),
	    tr("Fields such as ip.src_host then hold the names of the hosts, as the hosts files of "
	       "the system and the sources checked tell them. Asking DNS servers is slow, and seen "
	       "on the network."));

	_geoip = new QLineEdit;
	_geoip->setReadOnly(true);
	// A Flatpak sees those of the system only with access to the host.
	_geoip->setPlaceholderText(QFileInfo(pvpcap::DEFAULT_GEOIP_DIRECTORY).isDir()
	                               ? tr("those of the system")
	                               : tr("none: choose their directory"));
	auto* browse = new QToolButton;
	browse->setIcon(PVModdedIcon("folder-open"));
	browse->setToolTip(tr("Choose the directory of the MaxMind databases"));
	browse->setAutoRaise(true);
	auto* reset = new QToolButton;
	reset->setIcon(PVModdedIcon("arrow-rotate-left"));
	reset->setToolTip(tr("Use the databases of the system"));
	reset->setAutoRaise(true);
	_maxmind = new QLabel;
	_maxmind->setOpenExternalLinks(true);
	auto* geoip = new QHBoxLayout;
	geoip->addWidget(_geoip, 1);
	geoip->addWidget(browse);
	geoip->addWidget(reset);
	geoip->addWidget(_maxmind);
	add(tr("GeoIP databases"), geoip,
	    tr("Fields such as ip.geoip.src_country come from the MaxMind databases of this "
	       "directory."));
	grid->setRowStretch(row, 1);

	for (QCheckBox* box : {_reassemble, _defragment, _resolve, _from_captures, _from_servers}) {
		connect(box, &QCheckBox::toggled, this, &OptionsPanel::edit);
	}
	connect(_occurrence, &QComboBox::currentIndexChanged, this, &OptionsPanel::edit);
	connect(_wireshark_profile, &QComboBox::currentIndexChanged, this, &OptionsPanel::edit);
	connect(_aggregator, &QLineEdit::textEdited, this, &OptionsPanel::edit);
	connect(browse, &QToolButton::clicked, this, [this]() {
		const QString directory = PVWidgets::PVFileDialog::getExistingDirectory(
		    this, tr("Directory of the MaxMind databases"),
		    _geoip->text().isEmpty()
		        ? QStandardPaths::writableLocation(QStandardPaths::HomeLocation)
		        : _geoip->text());
		if (not directory.isEmpty()) {
			_geoip->setText(directory);
			edit();
		}
	});
	connect(reset, &QToolButton::clicked, this, [this]() {
		_geoip->clear();
		edit();
	});

	update_links();
	connect(&PVCore::PVTheme::get(), &PVCore::PVTheme::color_scheme_changed, this,
	        &OptionsPanel::update_links);
	set_profile({});
}

void OptionsPanel::set_profile(const pvpcap::profile& p)
{
	_showing = true;
	_profile = p;
	_reassemble->setChecked(p.reassemble_tcp);
	_defragment->setChecked(p.defragment_ip);
	_occurrence->setCurrentIndex(std::max(0, _occurrence->findData(QString(QChar(p.occurrence)))));
	_aggregator->setText(QString::fromStdString(p.aggregator));
	_aggregator->setEnabled(p.occurrence == 'a');
	_resolve->setChecked(p.resolve_names);
	_from_captures->setChecked(p.names_from_captures);
	_from_servers->setChecked(p.names_from_servers);
	_from_captures->setEnabled(p.resolve_names);
	_from_servers->setEnabled(p.resolve_names);
	_geoip->setText(QString::fromStdString(p.geoip_directory));

	// Only the profile shown may name a Wireshark profile Wireshark lacks.
	while (_wireshark_profile->count() > _wireshark_profiles) {
		_wireshark_profile->removeItem(_wireshark_profile->count() - 1);
	}
	const QString wanted = QString::fromStdString(p.wireshark_profile);
	int index = _wireshark_profile->findData(wanted);
	if (index < 0) {
		_wireshark_profile->addItem(PVModdedIcon("circle-x"), wanted, wanted);
		index = _wireshark_profile->count() - 1;
	}
	_wireshark_profile->setCurrentIndex(index);
	_wireshark_problem->setText(pvpcap::wireshark_profile_exists(p.wireshark_profile)
	                                ? QString()
	                                : tr("Wireshark has no profile named \"%1\": tshark reads "
	                                     "the captures without it.")
	                                      .arg(wanted));
	_wireshark_problem->setVisible(not _wireshark_problem->text().isEmpty());
	_showing = false;
}

QString OptionsPanel::problem() const
{
	return _wireshark_problem->text();
}

void OptionsPanel::edit()
{
	if (_showing) {
		return;
	}
	pvpcap::profile p = _profile;
	p.reassemble_tcp = _reassemble->isChecked();
	p.defragment_ip = _defragment->isChecked();
	const QString occurrence = _occurrence->currentData().toString();
	p.occurrence = occurrence.isEmpty() ? 'l' : occurrence.at(0).toLatin1();
	p.aggregator = _aggregator->text().isEmpty() ? "|" : _aggregator->text().toStdString();
	p.resolve_names = _resolve->isChecked();
	p.names_from_captures = _from_captures->isChecked();
	p.names_from_servers = _from_servers->isChecked();
	p.geoip_directory = _geoip->text().toStdString();
	p.wireshark_profile = _wireshark_profile->currentData().toString().toStdString();
	if (p != _profile) {
		_aggregator->setEnabled(p.occurrence == 'a');
		_from_captures->setEnabled(p.resolve_names);
		_from_servers->setEnabled(p.resolve_names);
		_profile = p;
		Q_EMIT edited(p);
	}
}

void OptionsPanel::update_links()
{
	_maxmind->setText(style::link("https://dev.maxmind.com/geoip/geolite2-free-geolocation-data",
	                              tr("Free databases")));
}

} // namespace PVPcapsicum
