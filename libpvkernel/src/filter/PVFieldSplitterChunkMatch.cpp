/**
 * \file PVFieldSplitterChunkMatch.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/filter/PVFieldSplitterChunkMatch.h>
#include <pvkernel/rush/PVRawSourceBase.h>
#include <pvkernel/core/picviz_bench.h>

#include <QHash>

#include <iostream>

#define GUESS_PVELEMENT_SAMPLE_NUMBER 20000

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
	typedef QHash<size_t, int>                            size_map_t;
	typedef std::pair<PVCore::PVArgumentList, size_map_t> data_t;
	typedef QHash<QString, data_t>                        data_map_t;

public:
	/**
	 * Add or increment the occurrence counter for the pair (al, nfields).
	 *
	 * @param[in] al the argument list to reference
	 * @param[in] nfields the fields number to reference
	 */
	void add(const PVCore::PVArgumentList &al, const size_t nfields)
	{
		if (nfields == 0) {
			return;
		}

		QString str;

		for (PVCore::PVArgumentList::const_iterator it = al.begin(); it != al.end(); ++it) {
			// use of a non-printable character as arguments values separator
			str.append("\01"+it->toString());
		}

		data_map_t::iterator it = _data_map.find(str);
		if (it == _data_map.end()) {
			data_t &d = _data_map[str];
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
		for (data_map_t::const_iterator dm_it = _data_map.begin(); dm_it != _data_map.end(); ++dm_it) {
			std::cout << "key: " << qPrintable(dm_it.key()) << std::endl << "  al :";

			for (PVCore::PVArgumentList::const_iterator al_it = dm_it->first.begin(); al_it != dm_it->first.end(); ++al_it) {
				std::cout << " (" << qPrintable(al_it->toString()) << ")";
			}
			std::cout << std::endl << "  cnt:";

			for (size_map_t::const_iterator sm_it = dm_it->second.begin(); sm_it != dm_it->second.end(); ++sm_it) {
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
	bool get_highest_entry(PVCore::PVArgumentList &al, size_t &nfields) const
	{
		bool ret = false;
		int highest = -1;

		for (data_map_t::const_iterator dm_it = _data_map.begin(); dm_it != _data_map.end(); ++dm_it) {
			for (size_map_t::const_iterator sm_it = dm_it->second.begin(); sm_it != dm_it->second.end(); ++sm_it) {
				if (highest < *sm_it) {
					ret = true;
					al = dm_it->first;
					nfields = sm_it.key();
					highest = *sm_it;
				}
			}
		}

		return ret;
	}

	size_t size() const
	{
		size_t s = 0;
		for (data_map_t::const_iterator dm_it = _data_map.begin(); dm_it != _data_map.end(); ++dm_it) {
			s += dm_it->second.size();
		}

		return s;
	}

private:
	data_map_t _data_map;
};

}

void PVFilter::PVFieldSplitterChunkMatch::push_chunk(PVCore::PVChunk* chunk)
{
	PVFilter::PVFieldsSplitter_p sp = _filter;
	PVCore::list_fields lf_res;
	PVCore::PVArgumentList args_match;

	PVCore::list_elts const& le = chunk->c_elements();
	PVCore::list_elts::const_iterator it_elt;
	size_t count = 0;
	BENCH_START(guessing);
	for (it_elt = le.begin(); it_elt != le.end(); it_elt++) {
		PVCore::PVField const& first_f = (*it_elt)->c_fields().front();
		sp->guess(_guess_res, first_f);
		++count;
		if (count > GUESS_PVELEMENT_SAMPLE_NUMBER) {
			break;
		}
	}
	BENCH_END(guessing, "processing chunk's elements", le.size(), 1, _guess_res.size(), 1);
}

bool PVFilter::PVFieldSplitterChunkMatch::get_match(PVCore::PVArgumentList& args, size_t& nfields)
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

#ifdef PICVIZ_DEVELOPER_MODE
	PVLOG_INFO("encountered splitting schemes\n");
	red.dump();
#endif

	return red.get_highest_entry(args, nfields);
}

PVFilter::PVFieldsSplitter_p PVFilter::PVFieldSplitterChunkMatch::get_match_on_input(PVRush::PVRawSourceBase_p src, PVCol &naxes)
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
	LIB_CLASS(PVFilter::PVFieldsSplitter)::list_classes const& lf = LIB_CLASS(PVFilter::PVFieldsSplitter)::get().get_list();
	LIB_CLASS(PVFilter::PVFieldsSplitter)::list_classes::const_iterator it;
	for (it = lf.begin(); it != lf.end(); it++) {
		PVFilter::PVFieldsSplitter_p sp = (*it)->clone<PVFilter::PVFieldsSplitter>();
		PVFilter::PVFieldSplitterChunkMatch match(sp);
		match.push_chunk(chunk);

		PVCore::PVArgumentList args;
		size_t nfields;

		if (match.get_match(args, nfields)) {
			PVLOG_DEBUG("(PVFieldSplitterChunkMatch) filter %s matches with %d fields\n with arguments:\n", qPrintable(it.key()), nfields);
			PVCore::dump_argument_list(args);
			ret = sp;
			ret->set_number_expected_fields(nfields);
			ret->set_args(args);
			naxes = nfields;
			break;
		}
	}
	chunk->free();

	return ret;
}
