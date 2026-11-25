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

#ifndef PCAP_GUI_CAPTURESPANEL_H
#define PCAP_GUI_CAPTURESPANEL_H

#include "Background.h"

#include <libpvpcap/sample.h>

#include <QString>
#include <QStringList>
#include <QWidget>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class QListWidget;
class QToolButton;

namespace PVPcapsicum
{

//! What is known of a capture to import, as it is read.
struct capture_info {
	QString path;
	std::optional<pvpcap::capture_overview> overview;
	//! Of its first frames: only of the first captures, which tell enough.
	std::optional<pvpcap::capture_sample> sample;
	//! What tshark said, if it can't read the capture.
	QString error;
	bool reading = false;
};

/**
 * The captures to import, in the order their frames are numbered in: added with a file
 * dialog, or dropped, and read at once to tell how large they are and what they hold.
 */
class CapturesPanel : public QWidget
{
	Q_OBJECT

  public:
	explicit CapturesPanel(QWidget* parent = nullptr);
	~CapturesPanel() override;

	//! Adds the files, and the captures in the directories, which are not there yet.
	void add(const QStringList& paths);
	//! Asks which captures to add.
	void ask();

	QStringList captures() const;
	//! In the order of captures().
	std::vector<const capture_info*> infos() const;

	/**
	 * How tshark reads the captures, which changes the fields it finds in them: the first
	 * frames are read again if they change.
	 */
	void set_reading_options(const std::vector<std::string>& options);

  Q_SIGNALS:
	void captures_changed();
	//! What is known of a capture changed.
	void infos_changed();
	//! The first frames of a capture were read by tshark, or no longer count.
	void samples_changed();

  private:
	struct entry {
		capture_info info;
		std::unique_ptr<BackgroundJob> job;
	};

	void read(const QString& path, bool overview);
	void update_item(const QString& path);
	void remove_selected();
	void captures_moved();
	//! Shows the list, or the zone that asks for captures while it is empty.
	void update_empty();

	QToolButton* _ask;
	QListWidget* _list;
	QToolButton* _remove;
	std::map<QString, entry> _entries;
	std::vector<std::string> _reading_options;
};

} // namespace PVPcapsicum

#endif // PCAP_GUI_CAPTURESPANEL_H
