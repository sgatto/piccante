/* header file for sselib2.c */

void PctAllocateDouble(double *pointer, int nsize);
void PctReallocateDouble(double *pointer, int newSize, int oldSize);

void sse_allocateDouble_128(double **s_f, int nsize, int *irc);
void sse_allocateDouble_256(double **s_f, int nsize, int *irc);

void sse_allocateFloat_128(float **s_f, int nsize, int *irc);
void sse_allocateFloat_256(float **s_f, int nsize, int *irc);

void sse_allocateInt_128(int **s_i, int nsize, int *irc);
void sse_allocateInt_256(int **s_i, int nsize, int *irc);

void sse_free(void *s_d);

void csse2iscan2(int *isdata, int nths);
