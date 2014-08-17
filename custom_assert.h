/*
	This header requires stdio or iostream (stderr and fprintf) AND stdlib (exit)
*/ 

#ifndef CUSTOM_ASSERT_H__
#define CUSTOM_ASSERT_H__

extern "C" int fprintf (FILE *__restrict __stream, const char *__restrict __format, ...);
extern struct _IO_FILE *stderr;		// Standard error output stream. 
extern "C" void exit(int);




#define AssertRuntime0(cond, msg, ...) { \
if (cond) { \
fprintf(stderr, "\n\nAssertion failed: "); \
fprintf(stderr, msg, ##__VA_ARGS__); \
fprintf(stderr, "\n\n\tFile: %s\n\tFunction: %s\n\tLine: %d\n\n\n",__FILE__,__FUNCTION__,__LINE__); \
exit(1); } \
}

#define AssertRuntime(cond, msg, ...) { AssertRuntime0(!(cond), msg, ##__VA_ARGS__); }

#define Guarantee  AssertRuntime
#define Guarantee0 AssertRuntime0

/*
#define AssertRuntime(cond, msg, ...) { if (! (cond)) { fprintf(stderr, "\n\nAssertion failed: ");fprintf(stderr,msg,##__VA_ARGS__);fprintf(stderr,"\n\n\tFile: %s\n\tFunction: %s\n\tLine: %d\n\n\n",__FILE__,__FUNCTION__,__LINE__); exit(1); }}
*/

// Only define the Assert versions if no debugging is enabled.
#ifndef NDEBUG
#define Assert0 AssertRuntime0
#define Assert  AssertRuntime 
#else
#define Assert0(cond, msg, ...) {}
#define Assert(cond, msg, ...) {}
#endif //NDEBUG



#endif// CUSTOM_ASSERT_H__
