#include <pvkernel/core/PVDecimalStorage.h>
#include <iostream>

typedef PVCore::PVDecimalStorage<32> decimal_t;

void f(size_t n, float* fr, float const* fa, float const* fb)
{
	for (size_t i = 0; i < n; i++) {
		fr[i] = fa[i] + fb[i];
	}
}

void f(size_t n, decimal_t* fr, decimal_t const* fa, decimal_t const* fb)
{
	for (size_t i = 0; i < n; i++) {
		fr[i].storage_as_float() = fa[i].storage_as_float() + fb[i].storage_as_float();
	}
}

struct holder
{
	template <typename T>
	static bool call(decimal_t const a, decimal_t const b)
	{
		return a.storage_cast<T>() < b.storage_cast<T>();
	}
};

int main(int argc, char** argv)
{
	PVCore::PVDecimalStorage<32> s;
	s.storage_as_int() = -4;

	std::cout << s.storage_cast<int>() << std::endl;
	std::cout << s.storage_cast<unsigned int>() << std::endl;

	s.storage_as_float() = 1.1f;
	std::cout << s.storage_cast<float>() << std::endl;

	// Vectorisation tests
	const size_t n = atoll(argv[1]);
	PVCore::PVDecimalStorage<32>* array_r = (PVCore::PVDecimalStorage<32>*) malloc(sizeof(unsigned int)*n);
	PVCore::PVDecimalStorage<32>* array_a = (PVCore::PVDecimalStorage<32>*) malloc(sizeof(unsigned int)*n);
	PVCore::PVDecimalStorage<32>* array_b = (PVCore::PVDecimalStorage<32>*) malloc(sizeof(unsigned int)*n);
	f(n, array_r, array_a, array_b);

	float *fr = &array_r[0].storage_as_float();
	float *fa = &array_a[0].storage_as_float();
	float *fb = &array_b[0].storage_as_float();
	f(n, fr, fa, fb);

	decimal_t da, db;
	da.storage_as_int() = 4;
	db.storage_as_int() = 5;
	std::cout << decimal_t::call_from_type<holder>(PVCore::IntegerType, da, db) << std::endl;

	da.set_max<float>();
	da.set_min<float>();
	std::cout << da.storage_as_float() << std::endl;

	return 0;
}
