/**
 * \file PVChunkTransform.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/PVChunkTransform.h>


size_t PVRush::PVChunkTransform::next_read_size(size_t org_size) const
{
	return org_size;
}

size_t PVRush::PVChunkTransform::operator()(char* /*data*/, size_t len_read, size_t /*len_avail*/) const
{
	// By default, no filtering/conversion
	return len_read;
}
