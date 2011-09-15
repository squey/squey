#include <pvkernel/rush/PVAggregator.h>
#include <pvkernel/rush/PVRawSourceBase.h>


PVRush::PVAggregator::PVAggregator(list_inputs const& inputs)
{
	_inputs = inputs;
	_src_offsets[0] = _inputs.begin();
	init();
}

PVRush::PVAggregator::PVAggregator()
{
	init();
}

void PVRush::PVAggregator::init()
{
	_eoi = false;
	_nstart = 0;
	_nlast = 0;
	_nend = pvconfig.value("pvkernel/extract_first", PVEXTRACT_NUMBER_LINES_FIRST_DEFAULT).toInt();
	__stop_cond_false = false;
	_stop_cond = &__stop_cond_false;
	_last_elt_agg_index = 0;
	_cur_input = _inputs.begin();
	_cur_src_index = 0;
	_strict_mode = false;
}


PVRush::PVAggregator::PVAggregator(const PVAggregator& /*org*/)
{
	assert(false);
}

void PVRush::PVAggregator::set_stop_condition(bool *cond)
{
	_stop_cond = cond;
}

void PVRush::PVAggregator::process_from_source(list_inputs::iterator input_start, chunk_index nstart, chunk_index nend)
{
	// Process from nstart to nend, starting by input_start
	
	// Find, compute offset for input_start
	if (!read_until_source(input_start)) {
		PVLOG_ERROR("(PVAggregator::process_from_source) unable to reach source %s. Using the last one...\n", qPrintable((*input_start)->human_name()));
		list_inputs::iterator it_last = _inputs.end();
		it_last--;
		if (input_start != it_last) {
			process_from_source(it_last, nstart, nend);
		}
		else {
			PVLOG_ERROR("(PVAggregator::process_from_source) already searching for the last source ! Starting from the beggining...\n");
			process_indexes(_nstart, _nend);
			return;
		}
	}
	chunk_index offset = _cur_src_index;

	// Then compute the new nstart and nend value
	_nstart = nstart + offset;
	_nend = nend + offset;

	// And use process_indexes
	process_indexes(_nstart, _nend);
}

void PVRush::PVAggregator::process_indexes(chunk_index nstart, chunk_index nend, chunk_index expected_nelts)
{
	_nstart = nstart;
	_nend = nend;
	_eoi = false;
	_last_elt_agg_index = 0;

	/*
	_nlast = 0;
	_last_elt_agg_index = 0;
	_cur_src_index = 0;
	_cur_input = _inputs.begin();

	list_inputs::iterator it;
	for (it = _inputs.begin(); it != _inputs.end(); it++) {
		// Reset all inputs position pointer
		PVLOG_DEBUG("PVExtractor::process_indexes seek begin on source %s\n", qPrintable((*it)->human_name()));
		(*it)->seek_begin();
	}*/

	if (expected_nelts == 0) {
		expected_nelts = nend-nstart;
	}

	// Find out the source that contains nstart
	
	chunk_index src_global_index = 0;
	list_inputs::iterator it_src = agg_index_to_source_iterator(nstart, &src_global_index);
	if (it_src == _inputs.end()) {
		// Unknown index, start from the beggining !
		_nlast = 0;
		_cur_src_index = 0;
		_cur_input = _inputs.begin();

		list_inputs::iterator it;
		for (it = _inputs.begin(); it != _inputs.end(); it++) {
			// Reset all inputs position pointer
			(*it)->seek_begin();
		}

		(*_cur_input)->prepare_for_nelts(expected_nelts);
		return;
	}

	// We found our source, now let's find the source offset of our index
	assert(nstart >= src_global_index);
	chunk_index src_index = nstart - src_global_index;
	chunk_index src_found_index;
	input_offset src_offset = (*it_src)->get_input_offset_from_index(src_index, src_found_index);

	// Set our aggregator accordingly
	_cur_input = it_src;
	if ((*_cur_input)->seek(src_offset)) {
		_cur_src_index = src_found_index;
		_nlast = src_global_index + src_found_index;
	}
	else {
		(*_cur_input)->seek_begin();
		_cur_src_index = 0;
		_nlast = src_global_index;
	}

	// Reset all inputs postion after where we are
	it_src++;
	for (; it_src != _inputs.end(); it_src++) {
		(*it_src)->seek_begin();
	}

	(*_cur_input)->prepare_for_nelts(expected_nelts);
}

bool PVRush::PVAggregator::read_until_source(list_inputs::iterator input_start)
{
	PVCore::PVChunk* c;
	while (_cur_input != input_start) {
		if ((c = next_chunk()) == NULL)
			return false;
		c->free();
	}
	return true;
}

PVCore::PVChunk* PVRush::PVAggregator::read_until_index(chunk_index idx) const
{
	PVCore::PVChunk* ret = NULL;
	while (_nlast < idx) {
		if (ret != NULL)
			ret->free();
		if ((ret = next_chunk()) == NULL)
			return NULL;
	}
	return ret;
}

PVCore::PVChunk* PVRush::PVAggregator::next_chunk() const
{
	if (*_stop_cond)
	{
		PVLOG_DEBUG("(PVAggregator::next_chunk) aggregator stop because of stop condition\n");
		return NULL;
	}

	// Get chunk from current input
	PVCore::PVChunk* ret = (*_cur_input)->operator()();

	while (ret == NULL) {
		// No more chunks available from current input. Go on !
		_cur_input++;
		if (_cur_input == _inputs.end()) {
			// No more data available. Return NULL
			return NULL;
		}
		_cur_src_index = _nlast;
		_src_offsets[_cur_src_index] = _cur_input;
		(*_cur_input)->prepare_for_nelts(_nend-_nlast);
		ret = (*_cur_input)->operator()();
	}

	if (ret != NULL) {
		ret->_agg_index = _cur_src_index + ret->_index;
		_nlast += ret->c_elements().size();
		_last_elt_agg_index = _nlast + ret->_index-1;
	}

	return ret;
}

PVCore::PVChunk* PVRush::PVAggregator::operator()() const
{
	if (_nlast >= _nend) {
		// This is the end of this job
		return NULL;
	}

	PVCore::PVChunk* ret;
	if (_nlast < _nstart) {
		// We have to read until _nstart indexes
		ret = read_until_index(_nstart);
	}
	else {
		ret = next_chunk();
		if (_strict_mode) {
			if (ret->_agg_index < _nstart) {
				chunk_index nelts = ret->c_elements().size();
				assert(ret->_agg_index + nelts - 1 >= _nstart);
				chunk_index nelts_remove = _nstart - ret->_agg_index;
				PVCore::list_elts& elts = ret->elements();
				PVCore::list_elts::iterator it_elt = elts.begin();
				for (chunk_index i = 0; i < nelts_remove; i++) {
					PVCore::list_elts::iterator it_er = it_elt;
					it_elt++;
					elts.erase(it_er);
				}
				ret->_agg_index += nelts_remove;
				ret->_index += nelts_remove;
			}
		}
	}

	if (ret == NULL) {
		_eoi = true;
		PVLOG_DEBUG("Aggregator: end of inputs\n");
	}

	return ret;
}

PVCore::PVChunk* PVRush::PVAggregator::operator()(tbb::flow_control &fc) const
{
	PVCore::PVChunk* ret = this->operator()();
	if (ret == NULL) {
		PVLOG_DEBUG("(PVAggregator::next_chunk) aggregator stop because of no more input datas\n");
		fc.stop();
	}
	return ret;
}

void PVRush::PVAggregator::read_all_chunks_from_beggining()
{
	PVLOG_DEBUG("(PVAggregator) read all chunks\n");
	// Just reset everything
	process_indexes(0, 1000);

	// Save the old stop condition pointer (just in case)
	bool* old_stop_cond = _stop_cond;
	// Never ever ever stop !!
	_stop_cond = &__stop_cond_false;
	// And here we go !!
	PVCore::PVChunk* chunk;
	while ((chunk = next_chunk()) != NULL) {
		chunk->free();
	}
	// Restore it
	_stop_cond = old_stop_cond;
}

PVRush::PVAggregator::list_inputs const& PVRush::PVAggregator::get_inputs() const
{
	return _inputs;
}

// Helper function
PVRush::PVAggregator_p PVRush::PVAggregator::from_unique_source(PVRush::PVRawSourceBase_p source)
{
	list_inputs inputs;
	inputs.push_back(source);

	return PVAggregator_p(new PVAggregator(inputs));
}

void PVRush::PVAggregator::add_input(PVRush::PVRawSourceBase_p in)
{
	_inputs.push_back(in);
	if (_inputs.size() == 1) {
		_src_offsets[0] = _inputs.begin();
	}
}

chunk_index PVRush::PVAggregator::last_elt_agg_index()
{
	return _last_elt_agg_index;
}

PVRush::PVRawSourceBase_p PVRush::PVAggregator::agg_index_to_source(chunk_index idx, chunk_index* global_index)
{
	list_inputs::iterator it_src = agg_index_to_source_iterator(idx, global_index);
	return (it_src != _inputs.end()) ? *it_src : PVRawSourceBase_p();
}

PVRush::PVAggregator::list_inputs::iterator PVRush::PVAggregator::agg_index_to_source_iterator(chunk_index idx, chunk_index* global_index)
{
	map_source_offsets::reverse_iterator it;
	for (it = _src_offsets.rbegin(); it != _src_offsets.rend(); it++) {
		if (idx >= (*it).first) {
			if (global_index)
				*global_index = (*it).first;
			return it->second;
		}
	}
	return _inputs.end();
}

void PVRush::PVAggregator::debug()
{
	// List sources
	PVLOG_DEBUG("PVAggregator::debug\n");
	list_inputs::iterator iti;
	PVLOG_DEBUG("PVAggregator::debug source\n");
	for (iti = _inputs.begin(); iti != _inputs.end(); iti++) {
		PVLOG_DEBUG("source %s\n", qPrintable((*iti)->human_name()));
	}

	PVLOG_DEBUG("PVAggregator::debug offset->source\n");
	map_source_offsets::iterator it;
	for (it = _src_offsets.begin(); it != _src_offsets.end(); it++)
		PVLOG_DEBUG("offset %d: source %s\n", (*it).first, qPrintable((*(it->second))->human_name()));
	PVLOG_DEBUG("PVAggregator::debug nstart=%d nlast=%d nend=%d\n", _nstart, _nlast, _nend);
}
