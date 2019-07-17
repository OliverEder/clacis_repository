/****************************************************************************
 * file - ip.h                                                              *
 *                                                                          *
 * For DOS compilers, the following pointers may need to be defined as huge *
 * or large depending on your compiler: image_ptr, pixel_ptr, double_ptr,   *
 * and complex_ptr.  IP_MALLOC and IP_FREE may also need to be redefined.   *
 * For the Borland Compiler, they should be defined as farmalloc and        *
 * and farfree.                                                             *
 *                                                                          *
 ****************************************************************************/

/* typedefs */

typedef unsigned char *image_ptr;
typedef double *double_ptr;
typedef struct
    {
    unsigned char r,g,b;
    } pixel;

typedef pixel *pixel_ptr;

typedef struct
    {
    int width;
    int height;
    float *x_data;
    float *y_data;
    } mesh;

typedef struct
    {
    double re;
    double im;
    } COMPLEX;

typedef COMPLEX *complex_ptr;

typedef struct
    {
    int x;
    int y;
    } POINT;

typedef struct 
    {
    POINT P;
    POINT Q;
    int dx, dy;
    float length;
    long length_squared;
    } LINE;

typedef struct 
    {
    POINT P;
    POINT Q;
    } LINE_SEGMENT;

typedef struct 
    {
    int number;             /* number of segments to follow */
    LINE_SEGMENT line[100];
    char *filename; /* name of file holding the line list */
    } LINE_LIST;

/* defines */

#define PI   3.14159265358979323846
#define CLIP(val, low, high) {if(val<low) val=low; if(val>high) val=high;}
#define CLAMP(val, low, high) ((val<low) ? low : ((val>high) ? high : val))
#define MAX(A,B)        ((A) > (B) ? (A) : (B))
#define MIN(A,B)        ((A) < (B) ? (A) : (B))
#define IP_MALLOC(X) malloc(X)
#define IP_FREE(X) free(X)
#define PBM 4
#define PGM 5
#define PPM 6

