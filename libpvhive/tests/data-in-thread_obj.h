
#ifndef entity_h
#define entity_h

class Entity
{
public:
	Entity(int i) : _i(i)
	{}

	void set_i(int i)
	{
		_i = i;
	}

	int get_i() const
	{
		return _i;
	}

private:
	int _i;
};

extern Entity *static_e;

#endif // entity_h
