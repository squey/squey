__kernel void bccb_naive(__global const unsigned int* codes, __global unsigned int* cb)
{
	int gid = get_global_id(0);
	const unsigned int c = codes[gid];
	atomic_or(&cb[c>>5], 1<<(c&31));
}

__kernel void bccb2(__global const unsigned int* codes, __global unsigned int* cb, __local unsigned int* local_cb)
{
	int gid = get_global_id(0);

	const unsigned int c = codes[gid];
	atomic_or(&cb[c>>5], 1<<(c&31));
}
