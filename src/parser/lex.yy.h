#ifndef das_yyHEADER_H
#define das_yyHEADER_H 1
#define das_yyIN_HEADER 1

#line 5 "lex.yy.h"

#line 7 "lex.yy.h"

#define  YY_INT_ALIGNED short int

/* A lexical scanner generated by flex */

#define FLEX_SCANNER
#define YY_FLEX_MAJOR_VERSION 2
#define YY_FLEX_MINOR_VERSION 6
#define YY_FLEX_SUBMINOR_VERSION 4
#if YY_FLEX_SUBMINOR_VERSION > 0
#define FLEX_BETA
#endif

#ifdef yy_create_buffer
#define das_yy_create_buffer_ALREADY_DEFINED
#else
#define yy_create_buffer das_yy_create_buffer
#endif

#ifdef yy_delete_buffer
#define das_yy_delete_buffer_ALREADY_DEFINED
#else
#define yy_delete_buffer das_yy_delete_buffer
#endif

#ifdef yy_scan_buffer
#define das_yy_scan_buffer_ALREADY_DEFINED
#else
#define yy_scan_buffer das_yy_scan_buffer
#endif

#ifdef yy_scan_string
#define das_yy_scan_string_ALREADY_DEFINED
#else
#define yy_scan_string das_yy_scan_string
#endif

#ifdef yy_scan_bytes
#define das_yy_scan_bytes_ALREADY_DEFINED
#else
#define yy_scan_bytes das_yy_scan_bytes
#endif

#ifdef yy_init_buffer
#define das_yy_init_buffer_ALREADY_DEFINED
#else
#define yy_init_buffer das_yy_init_buffer
#endif

#ifdef yy_flush_buffer
#define das_yy_flush_buffer_ALREADY_DEFINED
#else
#define yy_flush_buffer das_yy_flush_buffer
#endif

#ifdef yy_load_buffer_state
#define das_yy_load_buffer_state_ALREADY_DEFINED
#else
#define yy_load_buffer_state das_yy_load_buffer_state
#endif

#ifdef yy_switch_to_buffer
#define das_yy_switch_to_buffer_ALREADY_DEFINED
#else
#define yy_switch_to_buffer das_yy_switch_to_buffer
#endif

#ifdef yypush_buffer_state
#define das_yypush_buffer_state_ALREADY_DEFINED
#else
#define yypush_buffer_state das_yypush_buffer_state
#endif

#ifdef yypop_buffer_state
#define das_yypop_buffer_state_ALREADY_DEFINED
#else
#define yypop_buffer_state das_yypop_buffer_state
#endif

#ifdef yyensure_buffer_stack
#define das_yyensure_buffer_stack_ALREADY_DEFINED
#else
#define yyensure_buffer_stack das_yyensure_buffer_stack
#endif

#ifdef yylex
#define das_yylex_ALREADY_DEFINED
#else
#define yylex das_yylex
#endif

#ifdef yyrestart
#define das_yyrestart_ALREADY_DEFINED
#else
#define yyrestart das_yyrestart
#endif

#ifdef yylex_init
#define das_yylex_init_ALREADY_DEFINED
#else
#define yylex_init das_yylex_init
#endif

#ifdef yylex_init_extra
#define das_yylex_init_extra_ALREADY_DEFINED
#else
#define yylex_init_extra das_yylex_init_extra
#endif

#ifdef yylex_destroy
#define das_yylex_destroy_ALREADY_DEFINED
#else
#define yylex_destroy das_yylex_destroy
#endif

#ifdef yyget_debug
#define das_yyget_debug_ALREADY_DEFINED
#else
#define yyget_debug das_yyget_debug
#endif

#ifdef yyset_debug
#define das_yyset_debug_ALREADY_DEFINED
#else
#define yyset_debug das_yyset_debug
#endif

#ifdef yyget_extra
#define das_yyget_extra_ALREADY_DEFINED
#else
#define yyget_extra das_yyget_extra
#endif

#ifdef yyset_extra
#define das_yyset_extra_ALREADY_DEFINED
#else
#define yyset_extra das_yyset_extra
#endif

#ifdef yyget_in
#define das_yyget_in_ALREADY_DEFINED
#else
#define yyget_in das_yyget_in
#endif

#ifdef yyset_in
#define das_yyset_in_ALREADY_DEFINED
#else
#define yyset_in das_yyset_in
#endif

#ifdef yyget_out
#define das_yyget_out_ALREADY_DEFINED
#else
#define yyget_out das_yyget_out
#endif

#ifdef yyset_out
#define das_yyset_out_ALREADY_DEFINED
#else
#define yyset_out das_yyset_out
#endif

#ifdef yyget_leng
#define das_yyget_leng_ALREADY_DEFINED
#else
#define yyget_leng das_yyget_leng
#endif

#ifdef yyget_text
#define das_yyget_text_ALREADY_DEFINED
#else
#define yyget_text das_yyget_text
#endif

#ifdef yyget_lineno
#define das_yyget_lineno_ALREADY_DEFINED
#else
#define yyget_lineno das_yyget_lineno
#endif

#ifdef yyset_lineno
#define das_yyset_lineno_ALREADY_DEFINED
#else
#define yyset_lineno das_yyset_lineno
#endif

#ifdef yyget_column
#define das_yyget_column_ALREADY_DEFINED
#else
#define yyget_column das_yyget_column
#endif

#ifdef yyset_column
#define das_yyset_column_ALREADY_DEFINED
#else
#define yyset_column das_yyset_column
#endif

#ifdef yywrap
#define das_yywrap_ALREADY_DEFINED
#else
#define yywrap das_yywrap
#endif

#ifdef yyget_lval
#define das_yyget_lval_ALREADY_DEFINED
#else
#define yyget_lval das_yyget_lval
#endif

#ifdef yyset_lval
#define das_yyset_lval_ALREADY_DEFINED
#else
#define yyset_lval das_yyset_lval
#endif

#ifdef yyalloc
#define das_yyalloc_ALREADY_DEFINED
#else
#define yyalloc das_yyalloc
#endif

#ifdef yyrealloc
#define das_yyrealloc_ALREADY_DEFINED
#else
#define yyrealloc das_yyrealloc
#endif

#ifdef yyfree
#define das_yyfree_ALREADY_DEFINED
#else
#define yyfree das_yyfree
#endif

/* First, we deal with  platform-specific or compiler-specific issues. */

/* begin standard C headers. */
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

/* end standard C headers. */

/* flex integer type definitions */

#ifndef FLEXINT_H
#define FLEXINT_H

/* C99 systems have <inttypes.h>. Non-C99 systems may or may not. */

#if defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L

/* C99 says to define __STDC_LIMIT_MACROS before including stdint.h,
 * if you want the limit (max/min) macros for int types. 
 */
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS 1
#endif

#include <inttypes.h>
typedef int8_t flex_int8_t;
typedef uint8_t flex_uint8_t;
typedef int16_t flex_int16_t;
typedef uint16_t flex_uint16_t;
typedef int32_t flex_int32_t;
typedef uint32_t flex_uint32_t;
#else
typedef signed char flex_int8_t;
typedef short int flex_int16_t;
typedef int flex_int32_t;
typedef unsigned char flex_uint8_t; 
typedef unsigned short int flex_uint16_t;
typedef unsigned int flex_uint32_t;

/* Limits of integral types. */
#ifndef INT8_MIN
#define INT8_MIN               (-128)
#endif
#ifndef INT16_MIN
#define INT16_MIN              (-32767-1)
#endif
#ifndef INT32_MIN
#define INT32_MIN              (-2147483647-1)
#endif
#ifndef INT8_MAX
#define INT8_MAX               (127)
#endif
#ifndef INT16_MAX
#define INT16_MAX              (32767)
#endif
#ifndef INT32_MAX
#define INT32_MAX              (2147483647)
#endif
#ifndef UINT8_MAX
#define UINT8_MAX              (255U)
#endif
#ifndef UINT16_MAX
#define UINT16_MAX             (65535U)
#endif
#ifndef UINT32_MAX
#define UINT32_MAX             (4294967295U)
#endif

#ifndef SIZE_MAX
#define SIZE_MAX               (~(size_t)0)
#endif

#endif /* ! C99 */

#endif /* ! FLEXINT_H */

/* begin standard C++ headers. */

/* TODO: this is always defined, so inline it */
#define yyconst const

#if defined(__GNUC__) && __GNUC__ >= 3
#define yynoreturn __attribute__((__noreturn__))
#else
#define yynoreturn
#endif

/* An opaque pointer. */
#ifndef YY_TYPEDEF_YY_SCANNER_T
#define YY_TYPEDEF_YY_SCANNER_T
typedef void* yyscan_t;
#endif

/* For convenience, these vars (plus the bison vars far below)
   are macros in the reentrant scanner. */
#define yyin yyg->yyin_r
#define yyout yyg->yyout_r
#define yyextra yyg->yyextra_r
#define yyleng yyg->yyleng_r
#define yytext yyg->yytext_r
#define yylineno (YY_CURRENT_BUFFER_LVALUE->yy_bs_lineno)
#define yycolumn (YY_CURRENT_BUFFER_LVALUE->yy_bs_column)
#define yy_flex_debug yyg->yy_flex_debug_r

/* Size of default input buffer. */
#ifndef YY_BUF_SIZE
#ifdef __ia64__
/* On IA-64, the buffer size is 16k, not 8k.
 * Moreover, YY_BUF_SIZE is 2*YY_READ_BUF_SIZE in the general case.
 * Ditto for the __ia64__ case accordingly.
 */
#define YY_BUF_SIZE 32768
#else
#define YY_BUF_SIZE 16384
#endif /* __ia64__ */
#endif

#ifndef YY_TYPEDEF_YY_BUFFER_STATE
#define YY_TYPEDEF_YY_BUFFER_STATE
typedef struct yy_buffer_state *YY_BUFFER_STATE;
#endif

#ifndef YY_TYPEDEF_YY_SIZE_T
#define YY_TYPEDEF_YY_SIZE_T
typedef size_t yy_size_t;
#endif

#ifndef YY_STRUCT_YY_BUFFER_STATE
#define YY_STRUCT_YY_BUFFER_STATE
struct yy_buffer_state
	{
	FILE *yy_input_file;

	char *yy_ch_buf;		/* input buffer */
	char *yy_buf_pos;		/* current position in input buffer */

	/* Size of input buffer in bytes, not including room for EOB
	 * characters.
	 */
	int yy_buf_size;

	/* Number of characters read into yy_ch_buf, not including EOB
	 * characters.
	 */
	int yy_n_chars;

	/* Whether we "own" the buffer - i.e., we know we created it,
	 * and can realloc() it to grow it, and should free() it to
	 * delete it.
	 */
	int yy_is_our_buffer;

	/* Whether this is an "interactive" input source; if so, and
	 * if we're using stdio for input, then we want to use getc()
	 * instead of fread(), to make sure we stop fetching input after
	 * each newline.
	 */
	int yy_is_interactive;

	/* Whether we're considered to be at the beginning of a line.
	 * If so, '^' rules will be active on the next match, otherwise
	 * not.
	 */
	int yy_at_bol;

    int yy_bs_lineno; /**< The line count. */
    int yy_bs_column; /**< The column count. */

	/* Whether to try to fill the input buffer when we reach the
	 * end of it.
	 */
	int yy_fill_buffer;

	int yy_buffer_status;

	};
#endif /* !YY_STRUCT_YY_BUFFER_STATE */

void yyrestart ( FILE *input_file , yyscan_t yyscanner );
void yy_switch_to_buffer ( YY_BUFFER_STATE new_buffer , yyscan_t yyscanner );
YY_BUFFER_STATE yy_create_buffer ( FILE *file, int size , yyscan_t yyscanner );
void yy_delete_buffer ( YY_BUFFER_STATE b , yyscan_t yyscanner );
void yy_flush_buffer ( YY_BUFFER_STATE b , yyscan_t yyscanner );
void yypush_buffer_state ( YY_BUFFER_STATE new_buffer , yyscan_t yyscanner );
void yypop_buffer_state ( yyscan_t yyscanner );

YY_BUFFER_STATE yy_scan_buffer ( char *base, yy_size_t size , yyscan_t yyscanner );
YY_BUFFER_STATE yy_scan_string ( const char *yy_str , yyscan_t yyscanner );
YY_BUFFER_STATE yy_scan_bytes ( const char *bytes, int len , yyscan_t yyscanner );

void *yyalloc ( yy_size_t , yyscan_t yyscanner );
void *yyrealloc ( void *, yy_size_t , yyscan_t yyscanner );
void yyfree ( void * , yyscan_t yyscanner );

/* Begin user sect3 */

#define das_yywrap(yyscanner) (/*CONSTCOND*/1)
#define YY_SKIP_YYWRAP

#define yytext_ptr yytext_r

#ifdef YY_HEADER_EXPORT_START_CONDITIONS
#define INITIAL 0
#define indent 1
#define strb 2
#define strfmt 3
#define c_comment 4
#define cpp_comment 5
#define include 6
#define normal 7
#define reader 8

#endif

#ifndef YY_NO_UNISTD_H
/* Special case for "unistd.h", since it is non-ANSI. We include it way
 * down here because we want the user's section 1 to have been scanned first.
 * The user has a chance to override it with an option.
 */
#include <unistd.h>
#endif

#ifndef YY_EXTRA_TYPE
#define YY_EXTRA_TYPE void *
#endif

int yylex_init (yyscan_t* scanner);

int yylex_init_extra ( YY_EXTRA_TYPE user_defined, yyscan_t* scanner);

/* Accessor methods to globals.
   These are made visible to non-reentrant scanners for convenience. */

int yylex_destroy ( yyscan_t yyscanner );

int yyget_debug ( yyscan_t yyscanner );

void yyset_debug ( int debug_flag , yyscan_t yyscanner );

YY_EXTRA_TYPE yyget_extra ( yyscan_t yyscanner );

void yyset_extra ( YY_EXTRA_TYPE user_defined , yyscan_t yyscanner );

FILE *yyget_in ( yyscan_t yyscanner );

void yyset_in  ( FILE * _in_str , yyscan_t yyscanner );

FILE *yyget_out ( yyscan_t yyscanner );

void yyset_out  ( FILE * _out_str , yyscan_t yyscanner );

			int yyget_leng ( yyscan_t yyscanner );

char *yyget_text ( yyscan_t yyscanner );

int yyget_lineno ( yyscan_t yyscanner );

void yyset_lineno ( int _line_number , yyscan_t yyscanner );

int yyget_column  ( yyscan_t yyscanner );

void yyset_column ( int _column_no , yyscan_t yyscanner );

YYSTYPE * yyget_lval ( yyscan_t yyscanner );

void yyset_lval ( YYSTYPE * yylval_param , yyscan_t yyscanner );

/* Macros after this point can all be overridden by user definitions in
 * section 1.
 */

#ifndef YY_SKIP_YYWRAP
#ifdef __cplusplus
extern "C" int yywrap ( yyscan_t yyscanner );
#else
extern int yywrap ( yyscan_t yyscanner );
#endif
#endif

#ifndef yytext_ptr
static void yy_flex_strncpy ( char *, const char *, int , yyscan_t yyscanner);
#endif

#ifdef YY_NEED_STRLEN
static int yy_flex_strlen ( const char * , yyscan_t yyscanner);
#endif

#ifndef YY_NO_INPUT

#endif

/* Amount of stuff to slurp up with each read. */
#ifndef YY_READ_BUF_SIZE
#ifdef __ia64__
/* On IA-64, the buffer size is 16k, not 8k */
#define YY_READ_BUF_SIZE 16384
#else
#define YY_READ_BUF_SIZE 8192
#endif /* __ia64__ */
#endif

/* Number of entries by which start-condition stack grows. */
#ifndef YY_START_STACK_INCR
#define YY_START_STACK_INCR 25
#endif

/* Default declaration of generated scanner - a define so the user can
 * easily add parameters.
 */
#ifndef YY_DECL
#define YY_DECL_IS_OURS 1

extern int yylex \
               (YYSTYPE * yylval_param , yyscan_t yyscanner);

#define YY_DECL int yylex \
               (YYSTYPE * yylval_param , yyscan_t yyscanner)
#endif /* !YY_DECL */

/* yy_get_previous_state - get the state just before the EOB char was reached */

#undef YY_NEW_FILE
#undef YY_FLUSH_BUFFER
#undef yy_set_bol
#undef yy_new_buffer
#undef yy_set_interactive
#undef YY_DO_BEFORE_ACTION

#ifdef YY_DECL_IS_OURS
#undef YY_DECL_IS_OURS
#undef YY_DECL
#endif

#ifndef das_yy_create_buffer_ALREADY_DEFINED
#undef yy_create_buffer
#endif
#ifndef das_yy_delete_buffer_ALREADY_DEFINED
#undef yy_delete_buffer
#endif
#ifndef das_yy_scan_buffer_ALREADY_DEFINED
#undef yy_scan_buffer
#endif
#ifndef das_yy_scan_string_ALREADY_DEFINED
#undef yy_scan_string
#endif
#ifndef das_yy_scan_bytes_ALREADY_DEFINED
#undef yy_scan_bytes
#endif
#ifndef das_yy_init_buffer_ALREADY_DEFINED
#undef yy_init_buffer
#endif
#ifndef das_yy_flush_buffer_ALREADY_DEFINED
#undef yy_flush_buffer
#endif
#ifndef das_yy_load_buffer_state_ALREADY_DEFINED
#undef yy_load_buffer_state
#endif
#ifndef das_yy_switch_to_buffer_ALREADY_DEFINED
#undef yy_switch_to_buffer
#endif
#ifndef das_yypush_buffer_state_ALREADY_DEFINED
#undef yypush_buffer_state
#endif
#ifndef das_yypop_buffer_state_ALREADY_DEFINED
#undef yypop_buffer_state
#endif
#ifndef das_yyensure_buffer_stack_ALREADY_DEFINED
#undef yyensure_buffer_stack
#endif
#ifndef das_yylex_ALREADY_DEFINED
#undef yylex
#endif
#ifndef das_yyrestart_ALREADY_DEFINED
#undef yyrestart
#endif
#ifndef das_yylex_init_ALREADY_DEFINED
#undef yylex_init
#endif
#ifndef das_yylex_init_extra_ALREADY_DEFINED
#undef yylex_init_extra
#endif
#ifndef das_yylex_destroy_ALREADY_DEFINED
#undef yylex_destroy
#endif
#ifndef das_yyget_debug_ALREADY_DEFINED
#undef yyget_debug
#endif
#ifndef das_yyset_debug_ALREADY_DEFINED
#undef yyset_debug
#endif
#ifndef das_yyget_extra_ALREADY_DEFINED
#undef yyget_extra
#endif
#ifndef das_yyset_extra_ALREADY_DEFINED
#undef yyset_extra
#endif
#ifndef das_yyget_in_ALREADY_DEFINED
#undef yyget_in
#endif
#ifndef das_yyset_in_ALREADY_DEFINED
#undef yyset_in
#endif
#ifndef das_yyget_out_ALREADY_DEFINED
#undef yyget_out
#endif
#ifndef das_yyset_out_ALREADY_DEFINED
#undef yyset_out
#endif
#ifndef das_yyget_leng_ALREADY_DEFINED
#undef yyget_leng
#endif
#ifndef das_yyget_text_ALREADY_DEFINED
#undef yyget_text
#endif
#ifndef das_yyget_lineno_ALREADY_DEFINED
#undef yyget_lineno
#endif
#ifndef das_yyset_lineno_ALREADY_DEFINED
#undef yyset_lineno
#endif
#ifndef das_yyget_column_ALREADY_DEFINED
#undef yyget_column
#endif
#ifndef das_yyset_column_ALREADY_DEFINED
#undef yyset_column
#endif
#ifndef das_yywrap_ALREADY_DEFINED
#undef yywrap
#endif
#ifndef das_yyget_lval_ALREADY_DEFINED
#undef yyget_lval
#endif
#ifndef das_yyset_lval_ALREADY_DEFINED
#undef yyset_lval
#endif
#ifndef das_yyget_lloc_ALREADY_DEFINED
#undef yyget_lloc
#endif
#ifndef das_yyset_lloc_ALREADY_DEFINED
#undef yyset_lloc
#endif
#ifndef das_yyalloc_ALREADY_DEFINED
#undef yyalloc
#endif
#ifndef das_yyrealloc_ALREADY_DEFINED
#undef yyrealloc
#endif
#ifndef das_yyfree_ALREADY_DEFINED
#undef yyfree
#endif
#ifndef das_yytext_ALREADY_DEFINED
#undef yytext
#endif
#ifndef das_yyleng_ALREADY_DEFINED
#undef yyleng
#endif
#ifndef das_yyin_ALREADY_DEFINED
#undef yyin
#endif
#ifndef das_yyout_ALREADY_DEFINED
#undef yyout
#endif
#ifndef das_yy_flex_debug_ALREADY_DEFINED
#undef yy_flex_debug
#endif
#ifndef das_yylineno_ALREADY_DEFINED
#undef yylineno
#endif
#ifndef das_yytables_fload_ALREADY_DEFINED
#undef yytables_fload
#endif
#ifndef das_yytables_destroy_ALREADY_DEFINED
#undef yytables_destroy
#endif
#ifndef das_yyTABLES_NAME_ALREADY_DEFINED
#undef yyTABLES_NAME
#endif

#line 1131 "ds_lexer.lpp"


#line 726 "lex.yy.h"
#undef das_yyIN_HEADER
#endif /* das_yyHEADER_H */
