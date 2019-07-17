/*********************************************************************************************
* AUTOR: JESUS ANTONIO LUNA ALVAREZ                                                          *
* DESRIPCION: Apertura y lectura de archivos csv para visualizacion de datos lidar           *
*                                                                                            *
* ********************************************************************************************
* ORDEN DE COMPILACION                                                                       *
* g++ fusion.cpp -o fusion -lboost_iostreams -lboost_system -lboost_filesystem -std=c++11    *
*                                                                                            *
* ORDEN DE EJECUCION                                                                         *
* ./view3d                                                                                   *
*                                                                                            *
**********************************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "gnuplot-iostream.h"

int points = 50000;

int width = 1280;
int height = 640;
int CHANNEL_NUM = 3;

// Funcion que sirve para guardar los datos leidos del csv
void readFile(float *arr, char *dir){
  FILE *file = fopen(dir,"r");

	if(file == NULL)
    printf("\nError: file not found. \n\n");
  else{
    char *value = (char*) malloc(sizeof(char) * 1024);
    int i = 0, j = 0, row = 0;

    char chart;
    while((chart = fgetc(file)) != EOF){
      if(chart == '\n'){
        row++;
        j = -4;
      }

      if(row > 0){
        if(chart == ','){
          if(j >= 0 && j < points*3)
            *(arr+j) = atof(value);
          i = 0;
          j++;
          continue;
        }

        *(value+i) = chart;
        i++;
      }
    }
  }
  fclose(file);
}

// Funcion que sirve para guardar los datos leidos de la imagen
void readImg(float *img){
  int width, height, bpp;
  uint8_t* rgb_image = stbi_load("/home/tonny/bagfiles/2019-03-27-15-29-03/IMG/frame0000.jpg", &width, &height, &bpp, 3);

  for(int i=0;i<width*height*3;i++)
    *(img+i) = (float) *(rgb_image+i);
  stbi_image_free(rgb_image);
}

// Funcion que sirve para crear una nueva imagen
void writeImg(float * img){
  /*uint8_t* rgb_image = (uint8_t*) malloc(width*height*CHANNEL_NUM);
  for(int i=0;i<width*height*CHANNEL_NUM;i++)
    *(rgb_image+i) = 255 - (uint8_t) *(img+i);
  stbi_write_png("image.png", width, height, CHANNEL_NUM, rgb_image, width*CHANNEL_NUM);*/
  uint8_t* rgb_image = (uint8_t*) malloc(points*3);
  for(int i=2;i<points*3;i+=3){
    float v = ((*(img+i) - 1)/(50 - 1))*255.0;
    *(rgb_image+ (points*3-i)) = (uint8_t) 255.0 - v;
    printf("%d\t", *(rgb_image+i));
  }
  stbi_write_png("image.png", 500, 100, 3, rgb_image, 500*CHANNEL_NUM);
}

// invierte el eje Y de la nube de points
void trasopose(float *data){
  for(int i=0;i<points*3;i+=3){
    // ANTERIOR -> NUEVO
    // Y -> X
    // Z -> Y
    // X -> Z
    int z = *(data+i);
    *(data+i) = *(data+(i+1));
    *(data+(i+1)) = *(data+(i+2)) * -1;
    *(data+(i+2)) = z;
  }
}

// Desplicega grafica 3d de la nube de points
void graph(float *data, int limit){
  std::vector<std::tuple<float, float, float> > xy1;
	for(int i=0;i<limit;i+=3)
    xy1.push_back(std::make_tuple(data[i], data[i+1], data[i+2]));

  std::cout << "procesado" << '\n';

  Gnuplot gp;
  //gp << "set xrange [-2:2]\nset yrange [-2:2]\n";
  gp << "set autoscale" << std::endl;
  gp << "splot " << gp.file1d(xy1) << "with points title 'Cloud'" << std::endl;
  gp << "set xlabel 'Ax X'" << std::endl;
  gp << "set ylabel 'Ax Y'" << std::endl;
  gp << "set zlabel 'Ax Z'" << std::endl;
  std::cout << "graficado" << '\n';

}

int main(void){
  float *lidarData = (float*) malloc(sizeof(float) * points*3);
  readFile(lidarData, "/home/tonny/bagfiles/2019-03-27-15-29-03/lidar.csv");
  trasopose(lidarData);

  float *img = (float*) malloc(sizeof(float) * width*height*3);
  readImg(img);

  //writeImg(img);
  writeImg(lidarData);
  graph(lidarData, points*3);
  //graph(img, width*height*3);
	return 0;
}
