/*
 ============================================================================
 Name        : clacis.c
 Author      : Oliver Eder Guillermo III Espinosa Meneses
 Version     : 1.0
 Description : Competitive Learning Applied to Color Image Segmentation
 Compilation : gcc -o clacis_paralela -fopenmp clacis_paralela.c -lm 
 ============================================================================
 */
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>
#include "alloc.h"
#include "IPLIB.C"
#include <math.h>
#include <omp.h>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>

float **image;                          // Image pointer
float **neurons;                        // Neurons pointer 
image_ptr buffer;          				// pointer to image buffer  


extern void write_pnm(image_ptr ptr, char filein[], int rows, int cols, int magic_number);
extern image_ptr read_pnm(char *filename, int *rows, int *cols,int *type);

void normalize(int rows, int cols);
void set_number_neurons(int *number_neurons);
void set_neurons(int number_neurons);
void set_number_epoch(float *predefined_epoch);
void set_weight(int rows, int cols);
void set_image(int number_rgb);
void calculate_distances(  int rows, int cols, int selected_pattern, int *winner_neuron,float *min_distance);
void classifying( int rows, int cols, int selected_pattern, int *winner_neuron,float *min_distance );
void calculate_learning_rate(float learning_rate0, float *learning_rate, float predefined_epoch, float epoch);
void calculate_new_weights(int selected_pattern,int winner_neuron,float min_distance, float learning_rate);
void error(const char *s);
void get_Files( char *directory, char *files[]);
void show_neurons(int number_neurons);
void trainig_image(int number_rgb, int number_neurons);
void clasifying_image(int number_rgb, int number_neurons);
void denormalize(int number_rgb);
void training(int predefined_epoch, int training_files,int number_neurons,char directory);


// main function
int main(void)
{
    int training_files=6;
    int clasifying_files=236;  
    char directory[50]="./training";
    char *filesin[training_files];
    int rows, cols;                         // image rows and columns
    unsigned long bytes_per_pixel;      	// number of bytes per image pixel    
    unsigned long number_of_pixels;     	// total number of pixels in image
    int type;                           	// what type of image data
    struct timeval ti, tf;
    srand(time(NULL));
    double tiempo_entrenamiento=0, tiempo_clasificacion=0;
    int number_rgb=0;                       //number of pixels with the RGB1     
    int number_neurons=0;                   //number of neurons
    int selected_pattern=0;                 //ramdomly selected pattern
    int winner_neuron=0;                    //closest neuron to the pattern    
    float min_distance=0;                   //min distance tron de neuron to the pattern
    float learning_rate0=0.9;               //first learning rate
    float learning_rate=0.9;                //learning rate
    float predefined_epoch=0;               //Numero de epoacas predefinidas
       
    printf("========================================================\n");
	printf("Competitive Learning applied to Color Image Segmentation\n");
	printf("========================================================\n");
	set_number_neurons(&number_neurons);
	set_neurons(number_neurons);
    //set the weight values
    set_weight( number_neurons, 5);    
	set_number_epoch(&predefined_epoch);	
	printf("========================================================\n");
	printf("Training...\n" );
	printf("========================================================\n");
	printf("%f \n",predefined_epoch);
	int p_epoch=predefined_epoch;	                           
    // initial time
    gettimeofday(&ti, NULL);//Take te initial time
	for(int epoch=1 ; epoch <= p_epoch ; epoch++)
	{
        printf("Epoch %d\n",epoch);
        for(int i=0 ; i<training_files ; i++)
	    {
	        get_Files(&directory, &filesin);
	        char filein[100]="training/";           	// name of input file
	        strcat(filein,filesin[i]);
	        printf("%s\n",filein);	        
	        buffer = read_pnm(filein, &rows, &cols, &type);
	        bytes_per_pixel = 3;					   // 1 PGM IMAGES, 3 PPM IMAGES
	        number_of_pixels = bytes_per_pixel * rows * cols;
	        number_rgb=number_of_pixels/bytes_per_pixel;
            set_image(number_rgb);
	        normalize( number_rgb, 3);
	        trainig_image(number_rgb,number_neurons);	        
            free(image);
            image=NULL;
            IP_FREE(buffer);      
	    }    
	}
	gettimeofday(&tf, NULL);//Take de final time
    //final time
	tiempo_entrenamiento= (tf.tv_sec - ti.tv_sec)*1000 + (tf.tv_usec - ti.tv_usec)/1000.0;//calculates de time.
	
	/*Print the value of the neurons*/
	show_neurons(number_neurons);		                                        
    strcpy(directory,"./classifying");
    char *filesout[clasifying_files ];
    printf("========================================================\n");
    printf("Classifying...\n");
    printf("========================================================\n");
    // initial time
    gettimeofday(&ti, NULL);//Take te initial time
    for(int i=0 ; i<clasifying_files ; i++)
    {
        get_Files(&directory, &filesout);
        char fileout[100]="classifying/";          // name of input file
        strcat(fileout,filesout[i]);
        printf("%s\n",fileout);        
        char results[100]="results/";
        strcat(results,filesout[i]);        
        buffer = read_pnm(fileout, &rows, &cols, &type);
        bytes_per_pixel = 3;					// 1 PGM IMAGES, 3 PPM IMAGES
        number_of_pixels = bytes_per_pixel * rows * cols;
        number_rgb=number_of_pixels/bytes_per_pixel;        	        
        set_image(number_rgb);        
        normalize( number_rgb, 3);        
        clasifying_image(number_rgb, number_neurons);
        denormalize(number_rgb);                
        write_pnm(buffer,results, rows, cols ,6);
        free(image);
        image=NULL;       
        IP_FREE(buffer); 
    }
	
	gettimeofday(&tf, NULL);//Take de final time
    //final time
	tiempo_clasificacion= (tf.tv_sec - ti.tv_sec)*1000 + (tf.tv_usec - ti.tv_usec)/1000.0;//calculates de time.
	printf("\nTraining execution time: %g miliseconds\n", tiempo_entrenamiento);
	printf("Clasifycation execution time: %g miliseconds\n", tiempo_clasificacion);	
    return 0;
}


/*Normaliza los valores de los pixeles de la imagen, en un intervalo de 0 a 1*/
void normalize( int rows, int cols)
{
    int row = 0;
    int col = 0;
    #pragma omp paralel for 
    for(row = 0; row < rows; row++)
    {       
        for(col = 0; col < cols; col++)
        {
           *(*(image+row)+col)=*(*(image+row)+col)/255;           
        }                
    }        
}

/*Ingresar el numero de neuronas*/
void set_number_neurons(int *number_neurons)
{
    printf("Numero de neuronas:");
    scanf("%d",number_neurons);
} 

void set_neurons(int number_neurons)
{
    neurons= (float **) malloc(sizeof(float*) * number_neurons);
    for(int i = 0; i < number_neurons; i++)
    {
        neurons[i]=(float *) malloc(sizeof(float) * 5);
    }
}


/*Ingresar el numero de neuronas*/
void set_number_epoch(float *predefined_epoch)
{
    printf("Numero de epocas:");
    scanf("%f",predefined_epoch);
} 

void set_image(int number_rgb)
{
    int count=0;
    image = (float **) malloc(sizeof(float*) * number_rgb);	        
    for(int i = 0; i < number_rgb; i++)
    {
        image[i]=(float *) malloc(sizeof(float) * 3);
    }
    int row = 0;
    int channel = 0;
            
    #pragma omp paralel for 
    for(row = 0; row < number_rgb; row++)
    {
        for(channel = 0; channel < 3; channel++)
        {
            *(image[row]+channel)=buffer[count];
            count++;
        }
    }
        
}

/*Asigna un valor aleatorio entre 0 y 1 a los pesos de las neuronas*/
void set_weight( int rows, int cols)
{
    printf("Seting weights numbers...\n");
    int row = 0;
    int col = 0;    
    #pragma omp paralel for   
    for(row = 0; row < rows; row++) 
    {
        for(col = 0; col < 3; col++)
        {
           *(*(neurons+row)+col)=rand()%1000;
           *(*(neurons+row)+col)=*(*(neurons+row)+col)/1000;
           //printf("%f ",*(*(neurons+row)+col));
        }//printf("\n");                
    }    
        
}
/*
Calcula las distancias entre el patron de entrada y todas las neuronas
determina la neurona ganadora y debuelve la neurona ganadora y la distancia.
*/
void calculate_distances( int rows, int cols, int selected_pattern, int *winner_neuron,float *min_distance )
{
    float min=10000;
    int row = 0;
    int col = 0;
    //printf("Calculating distances...\n");
    #pragma omp paralel for       
    for(row = 0 ;row < rows; row++)
    {
        *(*(neurons+row)+4)=0;
    } 
    
    #pragma omp paralel for   
    for(row = 0; row < rows; row++)
    {                
        
        for(col = 0; col < 3; col++)
        {
           *(*(neurons+row)+4)=*(*(neurons+row)+4)+fabs(*(*(image+selected_pattern)+col) - *(*(neurons+row)+col));           
        }
        
        if(*(*(neurons+row)+4)<min)
        {
            min=*(*(neurons+row)+4);
            *winner_neuron=row;            
        }                
    }
    *min_distance=min;
    
    //printf("\nNeurona %d: %f\n", *winner_neuron , *min_distance);
}
/*
Calcula las distancias entre el patron de entrada y todas las neuronas
determina la neurona ganadora y debuelve la neurona ganadora y la distancia.
Clasifica al patron de entrada en el conjunto correspondiente a la neurona 
ganadora.
*/
void classifying( int rows, int cols, int selected_pattern, int *winner_neuron,float *min_distance )
{
    float min=10000;
    //printf("Classifying...\n");
   
    for(int row = 0; row < rows; row++)
    {
        
        for(int col = 0; col < 3; col++)
        {
           *(*(neurons+row)+4)=*(*(neurons+row)+4)+fabs(*(*(image+selected_pattern)+col) - *(*(neurons+row)+col));
           
        }        
        //printf("%f ",*(*(neurons+row)+4));
        if(*(*(neurons+row)+4)<min)
        {
            min=*(*(neurons+row)+4);
            *winner_neuron=row;
            *(*(image+selected_pattern)+3)=row;
        }                
    }
    *min_distance=min;
    //printf("\nNeurona %d: %f\n", *winner_neuron , *min_distance);
}
/*Clacula la taza de aprendizaje*/
void calculate_learning_rate(float learning_rate0, float *learning_rate, float predefined_epoch, float epoch)
{
    *learning_rate=learning_rate0*(1-(epoch/predefined_epoch));
}

/*Calcula los nuevos pesos de la neurona ganadora*/
void calculate_new_weights(int selected_pattern,int winner_neuron,float min_distance, float learning_rate)
{
    int col=0 ;
    if(min_distance!=0)
    {       
        //printf("w[k]%d      x[k]%d      w[k+1]%d\n", winner_neuron, selected_pattern,winner_neuron);
        #pragma omp paralel for
        for(col=0 ; col<3 ; col++)
        {
            *(*(neurons+winner_neuron)+col)=*(*(neurons+winner_neuron)+col)+(learning_rate*( *(*(image+selected_pattern)+col) - *(*(neurons+winner_neuron)+col) ));
        }
        //printf("\n");
        
    }
    //printf("============================================\n");
}
 
void error(const char *s)
{
  /* perror() devuelve la cadena S y el error (en cadena de caracteres) que tenga errno */
  perror (s);
  exit(EXIT_FAILURE);
}

void get_Files( char *directory, char *files[])
{
    int i=0;
    DIR *dir;
    /* en *ent habrá información sobre el archivo que se está "sacando" a cada momento */
    struct dirent *ent;
    /* Empezaremos a leer en el directorio actual */
    dir = opendir (directory);
    
    /* Miramos que no haya error */
    if (dir == NULL)
        error("No puedo abrir el directorio");
           
    /* Una vez nos aseguramos de que no hay error, ¡vamos a jugar! */
    /* Leyendo uno a uno todos los archivos que hay */
    while ((ent = readdir (dir)) != NULL)
    {
        // Nos devolverá el directorio actual (.) y el anterior (..), como hace ls 
        if ( (strcmp(ent->d_name, ".")!=0) && (strcmp(ent->d_name, "..")!=0) )
        {
          // Una vez tenemos el archivo, lo pasamos a una función para procesarlo. 
          files[i]=ent->d_name;
          i++;                    
        }
    }
    closedir (dir);    
}

void show_neurons(int number_neurons)
{
    int row = 0;
    int channel = 0;   
    for(row = 0; row < number_neurons; row++)
	{
        printf("%d  ",row );
        for(channel = 0; channel < 3; channel++)
        {
            printf("%f ", *(*(neurons+row)+channel) );
            
        }
        printf("\n");
	}
    printf("========================================================\n");
}

void write_image(int number_rgb, char results,int rows,int cols)
{
    int count=0;
    int row = 0;
    int channel = 0;
    int neuron=0;
    
    for(row = 0; row < number_rgb; row++)
    {
        neuron=*(image[row]+3);
        for(channel = 0; channel < 3; channel++)
        {
            buffer[count]=*(*(neurons+neuron)+channel)*255;
            count++;
        }            
    }        
    write_pnm(buffer,results, rows, cols ,6);
}

void trainig_image(int number_rgb, int number_neurons)
{
    int selected_pattern=0;                 //ramdomly selected pattern
    int winner_neuron=0;                    //closest neuron to the pattern    
    float min_distance=0;                   //min distance tron de neuron to the pattern
    float learning_rate0=0.9;               //first learning rate
    float learning_rate=0.9;                //learning rate
    int row = 0;
    #pragma omp paralel for
    for(row = 0; row < number_rgb; row++)
    {
        
        //calculate distances and get the closest neuron
        calculate_distances(  number_neurons, 5, row, &winner_neuron, &min_distance);
    
        //calculate new weights
        calculate_new_weights( row, winner_neuron, min_distance, learning_rate);
        
        //calculate learning rate
        calculate_learning_rate(learning_rate0, &learning_rate, number_rgb, row);                
    }
}

void clasifying_image(int number_rgb, int number_neurons)
{
    int selected_pattern=0;                 //ramdomly selected pattern
    int winner_neuron=0;                    //closest neuron to the pattern    
    float min_distance=0;                   //min distance tron de neuron to the pattern
    int row = 0;
    #pragma omp paralel for
    for(row = 0; row < number_rgb; row++)
    {
        //calculate distances and get the closest neuron
        calculate_distances( number_neurons, 5, row, &winner_neuron, &min_distance);
    
        //calculate distances and get the closest neuron
        classifying( number_neurons, 5, row, &winner_neuron, &min_distance);                
    }
}

void denormalize(int number_rgb)
{
    int count=0;
    int row = 0;
    int channel = 0;
    int neuron = 0;
    int newval=0;
    
          
    for(row = 0; row < number_rgb; row++)
    {
        neuron=*(image[row]+3);
        for(channel = 0; channel < 3; channel++)
        {
            buffer[count]=*(*(neurons+neuron)+channel)*255;
            count++;
        }            
    }    
}


