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

#ifndef LIBPVPCAP_PROFILE_H
#define LIBPVPCAP_PROFILE_H

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <rapidjson/document.h>

namespace pvpcap
{

struct profile_error : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

//! A field of tshark, with its type as tshark names it, such as FT_UINT16.
struct profile_field {
	std::string name;
	std::string type;

	bool operator==(const profile_field&) const = default;
};

/**
 * What to import of the frames of captures: the fields tshark writes of each, which make the
 * columns, and how tshark reads the captures.
 */
struct profile {
	//! In the order of the columns, each field once.
	std::vector<profile_field> columns;
	//! Display filter: the frames it drops are not imported. Empty to import all of them.
	std::string filter;
	/**
	 * What a field occurring more than once in a frame gives: its first value ('f'), its
	 * last one ('l'), or all of them ('a'), separated by `aggregator`.
	 */
	char occurrence = 'l';
	std::string aggregator = "|";
	bool reassemble_tcp = true;
	bool defragment_ip = true;
	//! Whether addresses are resolved, which fields such as ip.src_host then show.
	bool resolve_names = false;
	//! Where resolved names come from: the DNS answers the captures hold...
	bool names_from_captures = false;
	//! ...and the DNS servers of the system, which are then asked about every address.
	bool names_from_servers = false;
	//! Of the MaxMind databases the GeoIP fields come from, empty for the default one.
	std::string geoip_directory;
	//! Of Wireshark, whose preferences and "decode as" rules tshark follows: empty for none.
	std::string wireshark_profile;

	bool operator==(const profile&) const = default;
};

/**
 * Reads a profile, as this version writes it or as earlier ones did: those kept the tree of
 * the protocols of a capture, and the fields selected in it. Its columns are those fields, in
 * the order of the tree, then the columns of Wireshark it chose: Source, Destination,
 * Protocol and Info.
 *
 * @throw profile_error if it holds no profile
 */
profile profile_from_json(const rapidjson::Value& json);

//! Which earlier versions read too, as a profile selecting no field.
rapidjson::Document profile_to_json(const profile& p);

/**
 * The name tshark knows a field by, among those of the same field: tshark only writes such a
 * field at the last place it was asked, and leaves the others empty.
 *
 * The columns of the packet list of Wireshark keep the names they had in profiles, such as
 * _ws.col.Info rather than _ws.col.info.
 */
std::string canonical_field_name(const std::string& name);

//! A profile Squey ships, which can't be changed, or one of the user.
struct profile_location {
	bool builtin = false;
	std::string name;

	//! Of the file the profile is in.
	std::string path() const;

	bool operator==(const profile_location&) const = default;
};

//! The profiles Squey ships, then those of the user, each by name.
std::vector<profile_location> list_profiles();

//! @throw profile_error if it can't be read, saying why
profile read_profile(const std::string& path);

/**
 * Writes the whole profile, or leaves the file as it was.
 *
 * @throw profile_error if it can't be written, saying why
 */
void write_profile(const profile& p, const std::string& path);

/**
 * Why a profile of the user can't be given this name, which is the name of its file: empty
 * if it can, even if another profile has it.
 */
std::string profile_name_problem(const std::string& name);

//! Whether a profile of the user has this name, whatever the case of its letters.
bool user_profile_exists(const std::string& name);

//! `wanted`, followed by a number if a profile of the user already has that name.
std::string free_profile_name(const std::string& wanted);

//! Profiles changed and not saved yet, each with what it was changed to.
using profile_drafts = std::vector<std::pair<profile_location, profile>>;

//! The drafts write_drafts() kept, empty if there are none or they can't be read.
profile_drafts read_drafts();

/**
 * Keeps the drafts until they are read back, replacing those kept before.
 *
 * @throw profile_error if they can't be written, saying why
 */
void write_drafts(const profile_drafts& drafts);

} // namespace pvpcap

#endif // LIBPVPCAP_PROFILE_H
