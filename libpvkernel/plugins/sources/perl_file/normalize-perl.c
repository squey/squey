#include <stdio.h>

#include <picviz/general.h>
#include <picviz/debug.h>
#include <picviz/file.h>
#include <picviz/format.h>
#include <picviz/source.h>
#include <picviz/nraw.h>
#include <picviz/format.h>
#include <picviz/nraw.h>

#include <EXTERN.h>
#include <perl.h>


static PerlInterpreter *my_perl;

LibExport void picviz_normalize_init(void)
{
      my_perl = perl_alloc();
      perl_construct(my_perl);
}

LibExport char *picviz_normalize_discovery(char *filename)
{
	return NULL;
}

LibExport picviz_format_t *picviz_normalize_get_format(char *logopt)
{
	picviz_format_t *format;

	char *plugin_dir;
	char *args[2];
	char *ret_str;
	int retval;

	plugin_dir = picviz_plugins_get_normalize_helpers_path("perl");
	plugin_dir = realloc(plugin_dir, strlen(plugin_dir) + strlen(logopt) + strlen(".pl") + 1);
	
	sprintf(plugin_dir, "%s%s.pl", plugin_dir, logopt);

        args[0] = "";
	args[1] = plugin_dir;
	
	perl_parse(my_perl, NULL, 2, args, (char **)NULL);
        perl_run(my_perl);

	dSP;
        ENTER;
        SAVETMPS;
        PUSHMARK(SP);
        XPUSHs(sv_2mortal(newSVpv("NULL", 0))); /* Kept for later. We may pass a string */
        PUTBACK;
        retval = perl_call_pv("picviz_format", G_ARRAY);
        SPAGAIN;

	ret_str = POPp;
	/* Play with the stack now */
	/* printf("Ret str='%s'\n", ret_str); */
	format = picviz_format_create_from_buffer(ret_str);

        PUTBACK;
        FREETMPS;
        LEAVE;

	free(plugin_dir);

	return format;
}

LibExport picviz_nraw_t *picviz_normalize_file(picviz_source_t *source)
{
        char *string;
        AV *av;

        SV *sv;
        SV *array_sv;
        SV **array_elem_pp;
        SV *array_elem;

	int i,j;

	char *plugin_dir;
	char *args[2];
	char *ret_str;
	int retval;

	picviz_nraw_t *nraw;

	nraw = picviz_nraw_new(source->pool);
	if (!nraw) {
		picviz_debug(PICVIZ_DEBUG_CRITICAL, "Cannot create NRAW (%s)\n", __FUNCTION__);
		return NULL;
	}

	plugin_dir = picviz_plugins_get_normalize_helpers_path("perl");
	plugin_dir = realloc(plugin_dir, strlen(plugin_dir) + strlen(source->logopt) + strlen(".pl") + 1);
	
	sprintf(plugin_dir, "%s%s.pl", plugin_dir, source->logopt);

        args[0] = "";
	args[1] = plugin_dir;
	
	perl_parse(my_perl, NULL, 2, args, (char **)NULL);
        perl_run(my_perl);


	dSP;
	ENTER;
      	SAVETMPS;
      	PUSHMARK(SP);
        XPUSHs(sv_2mortal(newSVpv(source->file->filename, 0)));
      	PUTBACK;
      	retval = perl_call_pv("picviz_normalize", G_ARRAY);
      	SPAGAIN;


	for (i=0; i < retval; i++) {
		apr_array_header_t *nrawvalues = (apr_array_header_t *) picviz_nraw_values_array_new(nraw);
      	      sv = POPs;
      	      array_sv = SvRV(sv);
      	      av = (AV *)array_sv;

      	      /* printf("array len =%d\n", av_len(av)); */
	      /* New line */

      	      for (j=0;j<=av_len(av);j++) {
      		      array_elem_pp = av_fetch(av, j, FALSE);
      		      array_elem = *array_elem_pp;
      		      if (SvPOK(array_elem)) {
			/* New column */
			picviz_nraw_values_array_append(nraw, nrawvalues, SvPV_nolen(array_elem));
      			/* printf("value:%s\n", SvPV(array_elem, PL_na)); */
      			/* printf("value:%s\n", SvPV_nolen(array_elem)); */

      		      } else {
			picviz_debug(PICVIZ_DEBUG_CRITICAL, "The array at position [%d,%d] does not contain a string!\n", i, j);
      		      }
      	      }
		picviz_nraw_array_append(nraw, nrawvalues);


        }

        PUTBACK;
        FREETMPS;
        LEAVE; 

	free(plugin_dir);

	return nraw;
}

LibExport apr_array_header_t *picviz_normalize_buffer(picviz_source_t *source, char *buffer)
{
	return NULL;
}

LibExport void picviz_normalize_terminate(void)
{
      perl_destruct(my_perl);
      perl_free(my_perl);
}


