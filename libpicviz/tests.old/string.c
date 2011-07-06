#include <picviz/string.h>

int main(void)
{
        picviz_string_t string;
	picviz_string_t string2;
	picviz_string_t string3;
        /* safestr_t string; */
        /* safestr_t string2; */
	u_int32_t index;
	picviz_string_t string_noext;
	

	/* string = safestr_create("toto", SAFESTR_TEMPORARY); */
	/* string2 = safestr_create("tata", SAFESTR_TEMPORARY); */

	string = picviz_string_new("toto");
	/* string2 = picviz_string_new(" tata.pcre"); */
	
	/* picviz_string_append(string, string2); */
	picviz_string_append_buf(string, " tata.pcre");

	string3 = picviz_string_remove_extension(string);

	printf("end:'%s'\n", (char *)string3);

	/* index = safestr_rfindchar(string, '.'); */
	/* string_noext = safestr_slice(string, 0, index); */
	/* printf("str=%s;index=%d\n", (char *)string_noext, index); */
	
	picviz_string_destroy(string);
	/* picviz_string_destroy(string2); */

	return 0;
}
