#include <pvkernel/core/PVUtils.h>
#include <stdio.h>

int main(void)
{

	printf("Caps lock state:%d\n", PVCore::PVUtils::isCapsLockActivated());

	return 0;
}
