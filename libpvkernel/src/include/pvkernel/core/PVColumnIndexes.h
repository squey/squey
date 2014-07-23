/**
 * \file PVColumnArray.h
 *
 * Copyright (C) Picviz Labs 2014
 */
#ifndef __PVCORE_PVCOLUMNINDEXES_H__
#define __PVCORE_PVCOLUMNINDEXES_H__

namespace PVCore
{

class PVColumnIndexes {

public:
	typedef std::vector<PVCol> column_array_t;

public:
	inline PVCol operator[](size_t i) { return _indexes.at(i); }
	inline PVCol operator[](size_t i) const { return _indexes.at(i); }

	inline size_t size() const { return _indexes.size(); }
	inline void push_back(PVCol col) { _indexes.push_back(col); }

private:
	column_array_t _indexes;
};

}

#endif // __PVCORE_PVCOLUMNINDEXES_H__
