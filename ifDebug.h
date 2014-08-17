#ifndef DEBUG_H__
#define DEBUG_H__

#ifndef DNDEBUG
#define ifDebug(x) x
#else
#define ifDebug(x) {}
#endif

#endif // DEBUG_H__
