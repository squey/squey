#include <unistd.h>

int main(void)
{
	return execve("./variables.sh", NULL, NULL);
}
