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

#include "ProfileEditor.h"

#include <QFile>

#include <algorithm>

namespace PVPcapsicum
{

namespace
{

//! What a new profile starts with, which tells something of any frame.
const std::vector<pvpcap::profile_field> FIRST_COLUMNS = {{"frame.time", "FT_ABSOLUTE_TIME"},
                                                          {"_ws.col.Source", "FT_STRING"},
                                                          {"_ws.col.Destination", "FT_STRING"},
                                                          {"_ws.col.Protocol", "FT_STRING"},
                                                          {"frame.len", "FT_UINT32"}};

void check_new_name(const std::string& name, const std::string& unless)
{
	if (const std::string problem = pvpcap::profile_name_problem(name); not problem.empty()) {
		throw pvpcap::profile_error(problem);
	}
	// Renaming a profile changes no more than the case of its letters.
	const bool same = QString::fromStdString(name).compare(QString::fromStdString(unless),
	                                                       Qt::CaseInsensitive) == 0;
	if (not same and pvpcap::user_profile_exists(name)) {
		throw pvpcap::profile_error("A profile of yours already has this name.");
	}
}

} // namespace

ProfileEditor::ProfileEditor(QObject* parent) : QObject(parent) {}

void ProfileEditor::load(const pvpcap::profile_location& wanted)
{
	_profiles = pvpcap::list_profiles();
	_saved.clear();
	_drafts.clear();
	_problems.clear();
	for (const pvpcap::profile_location& location : _profiles) {
		read(location);
	}
	for (auto& [location, draft] : pvpcap::read_drafts()) {
		const auto saved = _saved.find(key_of(location));
		if (saved != _saved.end() and saved->second != draft) {
			_drafts[key_of(location)] = std::move(draft);
		}
	}
	const bool there = std::find(_profiles.begin(), _profiles.end(), wanted) != _profiles.end();
	_current = there or _profiles.empty() ? wanted : _profiles.front();
	Q_EMIT profiles_changed();
	Q_EMIT current_changed();
	Q_EMIT profile_changed();
}

void ProfileEditor::read(const pvpcap::profile_location& location)
{
	const key k = key_of(location);
	try {
		_saved[k] = pvpcap::read_profile(location.path());
		_problems.erase(k);
	} catch (const pvpcap::profile_error& e) {
		_saved[k] = pvpcap::profile();
		_problems[k] = QString::fromStdString(e.what());
	}
}

const pvpcap::profile& ProfileEditor::profile(const pvpcap::profile_location& location) const
{
	static const pvpcap::profile none;
	const key k = key_of(location);
	if (const auto draft = _drafts.find(k); draft != _drafts.end()) {
		return draft->second;
	}
	const auto saved = _saved.find(k);
	return saved != _saved.end() ? saved->second : none;
}

bool ProfileEditor::modified(const pvpcap::profile_location& location) const
{
	return _drafts.contains(key_of(location));
}

QString ProfileEditor::problem(const pvpcap::profile_location& location) const
{
	const auto it = _problems.find(key_of(location));
	return it != _problems.end() ? it->second : QString();
}

void ProfileEditor::choose(const pvpcap::profile_location& location)
{
	if (location == _current) {
		return;
	}
	_current = location;
	Q_EMIT current_changed();
	Q_EMIT profile_changed();
}

void ProfileEditor::change(const pvpcap::profile& changed)
{
	const key k = key_of(_current);
	const auto saved = _saved.find(k);
	if (saved != _saved.end() and saved->second == changed) {
		_drafts.erase(k);
	} else {
		_drafts[k] = changed;
	}
	Q_EMIT profile_changed();
}

void ProfileEditor::revert()
{
	_drafts.erase(key_of(_current));
	Q_EMIT profile_changed();
}

void ProfileEditor::save()
{
	if (_current.builtin) {
		throw pvpcap::profile_error("The profiles Squey ships can't be changed.");
	}
	const key k = key_of(_current);
	const pvpcap::profile p = profile();
	pvpcap::write_profile(p, _current.path());
	_saved[k] = p;
	_drafts.erase(k);
	_problems.erase(k);
	Q_EMIT profile_changed();
}

void ProfileEditor::add_user_profile(const std::string& name, const pvpcap::profile& p)
{
	check_new_name(name, {});
	const pvpcap::profile_location location{false, name};
	pvpcap::write_profile(p, location.path());
	_saved[key_of(location)] = p;
	_profiles = pvpcap::list_profiles();
	_current = location;
}

void ProfileEditor::save_as(const std::string& name)
{
	const key original = key_of(_current);
	add_user_profile(name, profile());
	_drafts.erase(original);
	Q_EMIT profiles_changed();
	Q_EMIT current_changed();
	Q_EMIT profile_changed();
}

void ProfileEditor::create(const std::string& name)
{
	pvpcap::profile p;
	p.columns = FIRST_COLUMNS;
	add_user_profile(name, p);
	Q_EMIT profiles_changed();
	Q_EMIT current_changed();
	Q_EMIT profile_changed();
}

void ProfileEditor::rename(const std::string& name)
{
	if (_current.builtin) {
		throw pvpcap::profile_error("The profiles Squey ships can't be renamed.");
	}
	if (name == _current.name) {
		return;
	}
	check_new_name(name, _current.name);
	const pvpcap::profile_location renamed{false, name};
	// Through another name: some file systems see no difference in the case of letters.
	const QString from = QString::fromStdString(_current.path());
	const QString between = from + ".renaming";
	const QString to = QString::fromStdString(renamed.path());
	if (not QFile::rename(from, between)) {
		throw pvpcap::profile_error("The file of the profile can't be renamed.");
	}
	if (not QFile::rename(between, to)) {
		QFile::rename(between, from);
		throw pvpcap::profile_error("The file of the profile can't be renamed.");
	}

	const key old_key = key_of(_current);
	const key new_key = key_of(renamed);
	for (auto* entries : {&_saved, &_drafts}) {
		if (auto node = entries->extract(old_key)) {
			node.key() = new_key;
			entries->insert(std::move(node));
		}
	}
	if (auto node = _problems.extract(old_key)) {
		node.key() = new_key;
		_problems.insert(std::move(node));
	}
	_profiles = pvpcap::list_profiles();
	_current = renamed;
	Q_EMIT profiles_changed();
	Q_EMIT current_changed();
}

void ProfileEditor::remove()
{
	if (_current.builtin) {
		throw pvpcap::profile_error("The profiles Squey ships can't be deleted.");
	}
	if (not QFile::remove(QString::fromStdString(_current.path()))) {
		throw pvpcap::profile_error("The file of the profile can't be deleted.");
	}
	const auto position = std::find(_profiles.begin(), _profiles.end(), _current);
	const size_t index = size_t(position - _profiles.begin());
	const key k = key_of(_current);
	_saved.erase(k);
	_drafts.erase(k);
	_problems.erase(k);
	_profiles = pvpcap::list_profiles();
	if (not _profiles.empty()) {
		// The one that took its place, or the one before if it was the last.
		_current = _profiles[std::min(index, _profiles.size() - 1)];
	}
	Q_EMIT profiles_changed();
	Q_EMIT current_changed();
	Q_EMIT profile_changed();
}

void ProfileEditor::import_file(const std::string& path, const std::string& name)
{
	add_user_profile(name, pvpcap::read_profile(path));
	Q_EMIT profiles_changed();
	Q_EMIT current_changed();
	Q_EMIT profile_changed();
}

void ProfileEditor::export_file(const std::string& path) const
{
	pvpcap::write_profile(profile(), path);
}

void ProfileEditor::write_drafts() const
{
	pvpcap::profile_drafts drafts;
	for (const auto& [k, p] : _drafts) {
		drafts.emplace_back(pvpcap::profile_location{k.first, k.second}, p);
	}
	pvpcap::write_drafts(drafts);
}

} // namespace PVPcapsicum
