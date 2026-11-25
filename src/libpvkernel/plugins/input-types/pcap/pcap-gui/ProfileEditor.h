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

#ifndef PCAP_GUI_PROFILEEDITOR_H
#define PCAP_GUI_PROFILEEDITOR_H

#include <libpvpcap/profile.h>

#include <QObject>
#include <QString>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace PVPcapsicum
{

/**
 * The profiles, and the one chosen, as changed: what the import dialog shows and changes.
 *
 * What is changed of a profile is a draft, kept until it is saved or reverted: also when
 * another profile is chosen, and once write_drafts() kept them for the next time. A profile
 * Squey ships can't be changed, only saved as a profile of the user.
 *
 * The functions changing files throw pvpcap::profile_error, saying why they could not.
 */
class ProfileEditor : public QObject
{
	Q_OBJECT

  public:
	explicit ProfileEditor(QObject* parent = nullptr);

	/**
	 * Reads the profiles, and the drafts write_drafts() kept: chooses `wanted`, or the first
	 * profile if there is no such profile.
	 */
	void load(const pvpcap::profile_location& wanted);

	//! Those Squey ships, then those of the user, each by name.
	const std::vector<pvpcap::profile_location>& profiles() const { return _profiles; }
	const pvpcap::profile_location& current() const { return _current; }
	//! The chosen profile, as changed.
	const pvpcap::profile& profile() const { return profile(_current); }
	const pvpcap::profile& profile(const pvpcap::profile_location& location) const;
	//! Whether the profile has changes not saved.
	bool modified(const pvpcap::profile_location& location) const;
	bool modified() const { return modified(_current); }
	//! Why the profile could not be read, empty if it was.
	QString problem(const pvpcap::profile_location& location) const;

	void choose(const pvpcap::profile_location& location);
	//! Changes the chosen profile.
	void change(const pvpcap::profile& changed);
	//! Drops the changes of the chosen profile.
	void revert();

	//! Saves the chosen profile, which is one of the user, as changed.
	void save();
	//! Saves the chosen profile as changed, as a new profile of the user, which is chosen.
	void save_as(const std::string& name);
	//! Creates a profile of the user with a few common columns, which is chosen.
	void create(const std::string& name);
	//! Renames the chosen profile, which is one of the user.
	void rename(const std::string& name);
	//! Deletes the chosen profile, which is one of the user, and its draft.
	void remove();
	//! Copies a file holding a profile as a new profile of the user, which is chosen.
	void import_file(const std::string& path, const std::string& name);
	//! Writes the chosen profile, as changed, to a file.
	void export_file(const std::string& path) const;

	//! Keeps the drafts for the next time the profiles are loaded.
	void write_drafts() const;

  Q_SIGNALS:
	//! Profiles came, went or were renamed.
	void profiles_changed();
	//! Another profile was chosen.
	void current_changed();
	//! The chosen profile changed, as it does when another one is chosen.
	void profile_changed();

  private:
	using key = std::pair<bool, std::string>;
	static key key_of(const pvpcap::profile_location& location)
	{
		return {location.builtin, location.name};
	}

	void read(const pvpcap::profile_location& location);
	void add_user_profile(const std::string& name, const pvpcap::profile& p);

	std::vector<pvpcap::profile_location> _profiles;
	pvpcap::profile_location _current;
	std::map<key, pvpcap::profile> _saved;
	std::map<key, pvpcap::profile> _drafts;
	std::map<key, QString> _problems;
};

} // namespace PVPcapsicum

#endif // PCAP_GUI_PROFILEEDITOR_H
