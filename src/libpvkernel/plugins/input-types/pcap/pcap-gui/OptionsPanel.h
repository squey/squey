/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PCAP_GUI_OPTIONSPANEL_H
#define PCAP_GUI_OPTIONSPANEL_H

#include <libpvpcap/profile.h>

#include <QWidget>

class QCheckBox;
class QComboBox;
class QLabel;
class QLineEdit;

namespace PVPcapsicum
{

/**
 * How tshark reads the captures, and what a column holds when its field repeats: the options
 * of the profile besides its columns and its filter, each explained where it is set.
 */
class OptionsPanel : public QWidget
{
	Q_OBJECT

  public:
	explicit OptionsPanel(QWidget* parent = nullptr);

	//! Shows the options of the profile, without telling they were edited.
	void set_profile(const pvpcap::profile& p);
	//! What makes tshark read the captures otherwise than the profile says, empty if nothing.
	QString problem() const;

  Q_SIGNALS:
	void edited(const pvpcap::profile& p);

  private:
	void edit();
	//! In the colors of the theme, which may change.
	void update_links();

	QCheckBox* _reassemble;
	QCheckBox* _defragment;
	QComboBox* _occurrence;
	QLineEdit* _aggregator;
	QComboBox* _wireshark_profile;
	//! Of _wireshark_profile, "none" and those of Wireshark, before one Wireshark lacks.
	int _wireshark_profiles = 0;
	QLabel* _wireshark_problem;
	QCheckBox* _resolve;
	QCheckBox* _from_captures;
	QCheckBox* _from_servers;
	QLineEdit* _geoip;
	QLabel* _maxmind;
	pvpcap::profile _profile;
	bool _showing = false;
};

} // namespace PVPcapsicum

#endif // PCAP_GUI_OPTIONSPANEL_H
