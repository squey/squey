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

#include <algorithm>
#include <vector>
#include <pvkernel/filter/PVFieldsFilter.h> // for PVFieldsSplitter_p
#include <pvkernel/core/PVArgument.h>     // for PVArgumentList
#include <pvkernel/core/PVTextChunk.h>    // for list_elts, PVChunk
#include <pvkernel/core/PVClassLibrary.h> // for LIB_CLASS
#include <pvkernel/core/PVElement.h>      // for list_fields, PVElement
#include <pvkernel/core/PVLogger.h>       // for PVLOG_INFO
#include <pvkernel/core/PVOrderedMap.h>
#include <pvkernel/core/squey_bench.h> // for BENCH_END, BENCH_START
#include <tbb/tick_count.h> // for tick_count
#include <assert.h>
#include <qtypeinfo.h>
#include <cstddef>  // for size_t
#include <iostream> // for operator<<, basic_ostream, etc
#include <list>     // for _List_iterator
#include <map>
#include <memory>   // for __shared_ptr, shared_ptr
#include <string_view>
#include <utility>  // for pair
#include <vector>   // for vector
#include <QHash>    // for QHash<>::const_iterator, etc
#include <QString>  // for QString, operator==, etc
#include <QVariant> // for QVariant

#include "pvbase/types.h" // for PVCol
#include "pvkernel/core/PVChunk.h"
#include "pvkernel/core/PVField.h"
#include "pvkernel/rush/PVRawSourceBase_types.h"

constexpr size_t GUESS_PVELEMENT_SAMPLE_NUMBER = 20000;

namespace PVFilter
{

/**
 * The values of @p al, which tell apart the argument lists of a splitter
 */
static QString key_of(const PVCore::PVArgumentList& al)
{
	QString str;
	for (const auto& it : al) {
		// use of a non-printable character as arguments values separator
		str.append("\01" + it.value().toString());
	}
	return str;
}

/**
 * @class PVGuessReducingTree
 *
 * Settles which of the argument lists a splitter guessed reads a sample of lines as
 * columns, and into how many.
 *
 * Each line the sample holds is added once per argument list that splits it into
 * several fields. An argument list stands for the number of fields most of the lines it
 * splits come apart into; it is dropped unless that number is regular enough and spans
 * enough of the sample, and the one kept splits the most lines into the most plausible
 * values. see PVFilter::PVFieldSplitterChunkMatch::get_match() for an example.
 */
class PVGuessReducingTree
{
	struct count_t {
		int lines = 0;
		double quality = 0;
		size_t fields = 0;
	};
	using size_map_t = std::map<size_t, count_t>;
	struct data_t {
		PVCore::PVArgumentList args;
		size_map_t counts;
		size_t rank = 0;
	};
	using data_map_t = QHash<QString, data_t>;

  public:
	/**
	 * Counts a record of @p lines lines the argument list @p al splits into @p nfields
	 * fields.
	 *
	 * @param quality how many of the fields are plausible values, from 0 to @p nfields
	 */
	void add(const PVCore::PVArgumentList& al, const size_t nfields, double quality, size_t lines)
	{
		if (nfields == 0) {
			return;
		}

		const QString str = key_of(al);
		data_map_t::iterator it = _data_map.find(str);
		if (it == _data_map.end()) {
			it = _data_map.insert(str, data_t{al, {}, size_t(_data_map.size())});
		}
		count_t& count = it->counts[nfields];
		count.lines += int(lines);
		count.quality += quality * double(lines);
		count.fields += nfields * lines;
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

			for (auto al_it = dm_it->args.begin(); al_it != dm_it->args.end(); ++al_it) {
				std::cout << " (" << qPrintable(al_it->value().toString()) << ")";
			}
			std::cout << std::endl << "  cnt:";

			for (const auto& [nfields, count] : dm_it->counts) {
				std::cout << " " << nfields << " (" << count.lines << ", quality "
				          << count.quality / count.fields << ")";
			}
			std::cout << std::endl;
		}
	}

	/**
	 * Keeps, for each argument list, the number of fields most of the lines it splits come
	 * apart into, and drops the argument lists for which that number is not regular.
	 *
	 * A few irregular lines do not rule an argument list out: a comment line or a total
	 * at the end of the file, a value holding the separator that nobody quoted. The
	 * splitter reads them anyway, padding the lines short of fields and leaving the excess
	 * in the last one.
	 *
	 * @param sampled_lines the number of lines of the sample, whether split or not
	 */
	void reduce(size_t sampled_lines)
	{
		data_map_t::iterator dm_it = _data_map.begin();
		while (dm_it != _data_map.end()) {
			int split_lines = 0;
			auto mode = dm_it->counts.begin();
			for (auto c_it = dm_it->counts.begin(); c_it != dm_it->counts.end(); ++c_it) {
				split_lines += c_it->second.lines;
				// of two numbers of fields as frequent, the larger
				if (c_it->second.lines >= mode->second.lines) {
					mode = c_it;
				}
			}
			const int regular_lines = mode->second.lines;
			if (regular_lines * 4 < split_lines * 3 or size_t(regular_lines) * 2 < sampled_lines) {
				dm_it = _data_map.erase(dm_it);
			} else {
				const auto kept = *mode;
				dm_it->counts = {kept};
				++dm_it;
			}
		}
	}

	/**
	 * Retrieve the entry that splits the most lines into the most plausible values.
	 *
	 * @param[out] al the found argument list
	 * @param[out] nfields  the found fields number
	 *
	 * @return true if an entry has been found; false otherwise.
	 */
	bool get_highest_entry(PVCore::PVArgumentList& al, PVCol& nfields) const
	{
		// Several argument lists are often regular on the same file: a comma-separated
		// file whose last field is a date splits just as evenly on the space. The cells
		// the space cuts out hold commas, though, and the plausibility of the values tells
		// the two apart. What is still even is settled by the number of fields told apart,
		// then by the order in which the splitter proposed its argument lists, as neither
		// may depend on the order of a hash table: Qt seeds it afresh on every run.
		struct candidate {
			double score;
			size_t fields;
			size_t rank;
			const PVCore::PVArgumentList* args;
		};

		std::vector<candidate> candidates;
		for (auto dm_it = _data_map.constBegin(); dm_it != _data_map.constEnd(); ++dm_it) {
			for (const auto& [fields, count] : dm_it->counts) {
				const double score = count.lines * (count.quality / count.fields);
				// not a line it reads with a plausible value
				if (score > 0) {
					candidates.push_back({score, fields, dm_it->rank, &dm_it->args});
				}
			}
		}

		if (candidates.empty()) {
			return false;
		}

		const auto& best = *std::min_element(
		    candidates.begin(), candidates.end(), [](const candidate& a, const candidate& b) {
			    if (a.score != b.score) {
				    return a.score > b.score;
			    }
			    if (a.fields != b.fields) {
				    return a.fields > b.fields;
			    }
			    return a.rank < b.rank;
		    });

		al = *best.args;
		nfields = PVCol(best.fields);

		return true;
	}

	size_t size() const
	{
		size_t s = 0;
		for (const auto& dm_it : _data_map) {
			s += dm_it.counts.size();
		}

		return s;
	}

  private:
	data_map_t _data_map;
};
} // namespace PVFilter

/**
 * How many lines push_chunk() counts in the sample that the record [@p begin, @p end) holds:
 * a record the splitter read across several lines stands for all of them.
 */
static size_t sampled_lines(const char* begin, const char* end)
{
	size_t lines = 0;
	for (const char* line = begin;; ++line) {
		const char* const lf = std::find(line, end, '\n');
		if (lf != line and not(lf == line + 1 and *line == '\r')) {
			++lines;
		}
		if (lf == end) {
			break;
		}
		line = lf;
	}
	return std::max<size_t>(lines, 1);
}

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
		if (first_f.size() != 0) {
			++_sampled_lines;
		}
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
		const PVCore::list_fields& fields = it->second;
		const size_t lines =
		    fields.empty() ? 1 : sampled_lines(fields.front().begin(), fields.back().end());
		red.add(it->first, fields.size(), _filter->guess_quality(it->first, fields), lines);
	}

	red.reduce(_sampled_lines);
	BENCH_END(reduction, "configuration detection", _guess_res.size(), 1, red.size(), 1);

#ifdef SQUEY_DEVELOPER_MODE
	PVLOG_INFO("encountered splitting schemes\n");
	red.dump();
#endif

	return red.get_highest_entry(args, nfields);
}

/**
 * How many lines @p chunk begins with that come before the records @p splitter reads into
 * @p nfields fields: see PVFilter::PVFieldSplitterChunkMatch::get_match_on_input().
 */
static size_t count_preamble_lines(PVCore::PVTextChunk& chunk,
                                   const PVFilter::PVFieldsSplitter& splitter,
                                   PVCol nfields)
{
	// past that many, the lines are what the file holds
	static constexpr size_t max_preamble_lines = 100;

	const QString args = PVFilter::key_of(splitter.get_args());
	// guess() changes the arguments of the splitter it is called on
	const PVFilter::PVFieldsSplitter_p probe = splitter.clone<PVFilter::PVFieldsSplitter>();
	size_t lines = 0;
	for (PVCore::PVElement* elt : chunk.elements()) {
		PVCore::PVField& line = elt->fields().front();
		const std::string_view text(line.begin(), line.size());
		if (lines == 0 and text.size() == 5 and text.starts_with("sep=")) {
			++lines;
			continue;
		}
		if (not text.starts_with('#')) {
			break;
		}
		// unless the '#' comments out the names of the columns, as in "#id,name"
		PVFilter::list_guess_result_t guesses;
		probe->guess(guesses, line);
		if (std::ranges::any_of(guesses, [&](const auto& guess) {
			    return guess.second.size() == size_t(nfields) and
			           PVFilter::key_of(guess.first) == args;
		    })) {
			break;
		}
		if (++lines == max_preamble_lines) {
			return 0;
		}
	}
	return lines;
}

PVFilter::PVFieldsSplitter_p PVFilter::PVFieldSplitterChunkMatch::get_match_on_input(
    PVRush::PVRawSourceBase_p src, PVCol& naxes, size_t* preamble_lines)
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
			if (preamble_lines != nullptr) {
				*preamble_lines = count_preamble_lines(*text_chunk, *ret, nfields);
			}
			break;
		}
	}
	text_chunk->free();

	return ret;
}
