//
// MIT License
//
// © ESI Group, 2015
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

#include <pvkernel/rush/PVRawSourceBase.h>

#include <pvkernel/filter/PVFieldSplitterChunkMatch.h>
#include <pvkernel/filter/PVFieldsFilter.h> // for PVFieldsSplitter_p

#include <pvkernel/core/PVArgument.h>     // for PVArgumentList
#include <pvkernel/core/PVTextChunk.h>    // for list_elts, PVChunk
#include <pvkernel/core/PVClassLibrary.h> // for LIB_CLASS
#include <pvkernel/core/PVElement.h>      // for list_fields, PVElement
#include <pvkernel/core/PVLogger.h>       // for PVLOG_INFO
#include <pvkernel/core/PVOrderedMap.h>
#include <pvkernel/core/inendi_bench.h> // for BENCH_END, BENCH_START

#include "pvbase/types.h" // for PVCol

#include <tbb/tick_count.h> // for tick_count

#include <cstddef>  // for size_t
#include <iostream> // for operator<<, basic_ostream, etc
#include <list>     // for _List_iterator
#include <memory>   // for __shared_ptr, shared_ptr
#include <utility>  // for pair
#include <vector>   // for vector

#include <QHash>    // for QHash<>::const_iterator, etc
#include <QString>  // for QString, operator==, etc
#include <QVariant> // for QVariant

namespace PVCore
{
class PVField;
} // namespace PVCore

constexpr size_t GUESS_PVELEMENT_SAMPLE_NUMBER = 20000;

namespace PVFilter
{

/**
 * @class PVGuessReducingTree
 *
 * This class is used for the search of a relevant splitter configuration when
 * detecting a format to use with a input.
 *
 * The normal use of this class is to insert all identified pair (argument list,
 * field number) to count their occurrences, reduce it to remove entries which
 * are not relevant and to get its entry which have the highest occurrence.
 * see PVFilter::PVFieldSplitterChunkMatch::get_match() for an example.
 */
class PVGuessReducingTree
{
	using size_map_t = QHash<size_t, int>;
	using data_t = std::pair<PVCore::PVArgumentList, size_map_t>;
	using data_map_t = QHash<QString, data_t>;

  public:
	/**
	 * Add or increment the occurrence counter for the pair (al, nfields).
	 *
	 * @param[in] al the argument list to reference
	 * @param[in] nfields the fields number to reference
	 */
	void add(const PVCore::PVArgumentList& al, const size_t nfields)
	{
		if (nfields == 0) {
			return;
		}

		QString str;

		for (const auto& it : al) {
			// use of a non-printable character as arguments values separator
			str.append("\01" + it.value().toString());
		}

		data_map_t::iterator it = _data_map.find(str);
		if (it == _data_map.end()) {
			data_t& d = _data_map[str];
			d.first = al;
			d.second[nfields] = 1;
		} else {
			it->second[nfields] += 1;
		}
	}

	/**
	 * Write the content to stdout.
	 *
	 * For debug purpose only.
	 */
	void dump() const
	{
		for (data_map_t::const_iterator dm_it = _data_map.begin(); dm_it != _data_map.end();
		     ++dm_it) {
			std::cout << "key: " << qPrintable(dm_it.key()) << std::endl << "  al :";

			for (auto al_it = dm_it->first.begin(); al_it != dm_it->first.end(); ++al_it) {
				std::cout << " (" << qPrintable(al_it->value().toString()) << ")";
			}
			std::cout << std::endl << "  cnt:";

			for (size_map_t::const_iterator sm_it = dm_it->second.begin();
			     sm_it != dm_it->second.end(); ++sm_it) {
				std::cout << " " << sm_it.key() << " (" << *sm_it << ")";
			}
			std::cout << std::endl;
		}
	}

	/**
	 * Remove entries which are not a stable splitting scheme (those for
	 * which a argument list have more than fields count).
	 */
	void reduce()
	{
		data_map_t::iterator dm_it = _data_map.begin();
		while (dm_it != _data_map.end()) {
			if (dm_it->second.size() > 1) {
				dm_it = _data_map.erase(dm_it);
			} else {
				++dm_it;
			}
		}
	}

	/**
	 * Retrieve the first entry with the highest occurrence number.
	 *
	 * @param[out] al the found argument list
	 * @param[out] nfields  the found fields number
	 *
	 * @return true if an entry has been found; false otherwise.
	 */
	bool get_highest_entry(PVCore::PVArgumentList& al, PVCol& nfields) const
	{
		bool ret = false;
		int highest = -1;

		for (const auto& dm_it : _data_map) {
			for (size_map_t::const_iterator sm_it = dm_it.second.begin();
			     sm_it != dm_it.second.end(); ++sm_it) {
				if (highest < *sm_it) {
					ret = true;
					al = dm_it.first;
					nfields = PVCol(sm_it.key());
					highest = *sm_it;
				}
			}
		}

		return ret;
	}

	size_t size() const
	{
		size_t s = 0;
		for (const auto& dm_it : _data_map) {
			s += dm_it.second.size();
		}

		return s;
	}

  private:
	data_map_t _data_map;
};
} // namespace PVFilter

void PVFilter::PVFieldSplitterChunkMatch::push_chunk(PVCore::PVChunk* c)
{
	auto* chunk = dynamic_cast<PVCore::PVTextChunk*>(c);
	assert(chunk);
	PVFilter::PVFieldsSplitter_p sp = _filter;
	PVCore::list_fields lf_res;
	PVCore::PVArgumentList args_match;

	PVCore::list_elts& le = chunk->elements();
	size_t count = 0;
	BENCH_START(guessing);
	for (auto& it_elt : le) {
		PVCore::PVField& first_f = it_elt->fields().front();
		sp->guess(_guess_res, first_f);
		++count;
		if (count > GUESS_PVELEMENT_SAMPLE_NUMBER) {
			break;
		}
	}
	BENCH_END(guessing, "processing chunk's elements", le.size(), 1, _guess_res.size(), 1);
}

bool PVFilter::PVFieldSplitterChunkMatch::get_match(PVCore::PVArgumentList& args, PVCol& nfields)
{
	// Reduce _guess_res and check which args always returns the same number of fields
	PVGuessReducingTree red;

	BENCH_START(reduction);
	PVFilter::list_guess_result_t::iterator it;

	for (it = _guess_res.begin(); it != _guess_res.end(); it++) {
		red.add(it->first, it->second.size());
	}

	red.reduce();
	BENCH_END(reduction, "configuration detection", _guess_res.size(), 1, red.size(), 1);

#ifdef INENDI_DEVELOPER_MODE
	PVLOG_INFO("encountered splitting schemes\n");
	red.dump();
#endif

	return red.get_highest_entry(args, nfields);
}

PVFilter::PVFieldsSplitter_p
PVFilter::PVFieldSplitterChunkMatch::get_match_on_input(PVRush::PVRawSourceBase_p src, PVCol& naxes)
{
	PVCore::PVChunk* chunk = (*src)();
	PVFieldsSplitter_p ret;
	if (!chunk) {
		src->seek_begin();
		chunk = (*src)();
		if (!chunk) {
			return ret;
		}
	}
	auto* text_chunk = dynamic_cast<PVCore::PVTextChunk*>(chunk);
	assert(text_chunk);
	LIB_CLASS(PVFilter::PVFieldsSplitter)
	::list_classes const& lf = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_list();
	LIB_CLASS(PVFilter::PVFieldsSplitter)::list_classes::const_iterator it;
	for (it = lf.begin(); it != lf.end(); it++) {
		PVFilter::PVFieldsSplitter_p sp = it->value()->clone<PVFilter::PVFieldsSplitter>();
		PVFilter::PVFieldSplitterChunkMatch match(sp);
		match.push_chunk(chunk);

		PVCore::PVArgumentList args;
		PVCol nfields;

		if (match.get_match(args, nfields)) {
			PVLOG_DEBUG(
			    "(PVFieldSplitterChunkMatch) filter %s matches with %d fields\n with arguments:\n",
			    qPrintable(it->key()), nfields);
			PVCore::dump_argument_list(args);
			ret = sp;
			ret->set_number_expected_fields(nfields);
			ret->set_args(args);
			naxes = nfields;
			break;
		}
	}
	text_chunk->free();

	return ret;
}
