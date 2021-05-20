

#include <stdio.h>

#define INSIZE 28
#define INFO_BYTE_SIZE 4

#define INITIAL_WEIGHT_VALUE -1.0f
#define INITIAL_FC_WEIGHT_VALUE 1.0f
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define CONV_FILTER 5
#define SS_FILTER 4
#define FEATURES 6 
#define NEURONS 10
#define CONV_OUTPUT 24
#define SS_OUTPUT 6
#define FC_OUTPUT 10

//kernel function that fill mnist_data structure->data with normalized pixel values
__global__ void fillArr(unsigned char pixels[INSIZE][INSIZE], double data[INSIZE][INSIZE]){
  // TO DO
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if(i<INSIZE && j<INSIZE) 
    data[i][j] = pixels[i][j]/255.0;
}

//kernel function that changes the values >0 to 1 and double type to integer type
__global__ void showArr(double ddata[INSIZE][INSIZE], int dshow[INSIZE][INSIZE]){
  // TO DO
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if(i<INSIZE && j<INSIZE){
    if(ddata[i][j]>0)
      dshow[i][j] = 1;
    else dshow[i][j] = 0;
  }
}

//mnist data structure
typedef struct mnist_data{
  double data[INSIZE][INSIZE];
  unsigned int label;
}mnist_data;

//structure for images header information
typedef struct images_info{
  char magic_num_images[INFO_BYTE_SIZE];
  char amount_images[INFO_BYTE_SIZE];
  char rows[INFO_BYTE_SIZE];
  char columns[INFO_BYTE_SIZE];
}images_info;

//structure for labels header information
typedef struct labels_info{
  char magic_num_labels[INFO_BYTE_SIZE];
  char amount_labels[INFO_BYTE_SIZE];
}labels_info;

//Hexadecimal to integer
static unsigned int mnist_bin_to_int(char *tmp){
    int val = (tmp[0] << 24 | tmp[1] << 16 | tmp[2] << 8 | tmp[3] );
    return val;
}

static int mnist_load(const char *image_filename, const char *label_filename, mnist_data **data_set,unsigned int *count){
                          
    images_info i_info;
    labels_info l_info;

    //opening the files
    FILE *images = fopen(image_filename,"rb");
    FILE *labels = fopen(label_filename,"rb");
    if(images==NULL||labels==NULL){
        return -1;
    }

    //read header info
    fread(&i_info,sizeof(images_info),1,images);
    fread(&l_info,sizeof(labels_info),1,labels);

    //check and print header info
    int magic_num_images_as_int = mnist_bin_to_int(i_info.magic_num_images);
    if(magic_num_images_as_int != 2051){
      printf("Problems with 'image magic number'. It is equal to %d, but should be 2051.",magic_num_images_as_int);
      return -1;
    }
    else{
      printf("image magic number = %d (should be 2051)\n", magic_num_images_as_int);   
    }

    int magic_num_labels_as_int = mnist_bin_to_int(l_info.magic_num_labels);
    if(magic_num_labels_as_int != 2049){
      printf("Problems with 'label magic number'. It is equal to %d, but should be 2049.",magic_num_labels_as_int);
      return -1;
    }
    else{
      printf("label magic number = %d (should be 2049)\n", magic_num_labels_as_int); 
    }

    int amount_images_as_int = mnist_bin_to_int(i_info.amount_images);
    if(amount_images_as_int != 10000){
      printf("Problems with 'image total number'. It is equal to %d, but should be 10000.",amount_images_as_int);
      return -1;
    }
    else{
      printf("image total number = %d (should be 10000)\n", amount_images_as_int); 
    }

    int amount_labels_as_int = mnist_bin_to_int(l_info.amount_labels);
    if(amount_labels_as_int != 10000){
      printf("Problems with 'label total number'. It is equal to %d, but should be 10000.",amount_labels_as_int);
      return -1;
    }
    else{
      printf("label total number = %d (should be 10000)\n", amount_labels_as_int);
    }

    int rows_as_int = mnist_bin_to_int(i_info.rows);
    int columns_as_int = mnist_bin_to_int(i_info.columns);
    if((rows_as_int != 28)||(columns_as_int!=28)){
      printf("Problems with dimensions of images. Dimensions of images are not compitable with 28x28.");
      return -1;
    }
    else{
      printf("rows = %d, cols = %d (both should be 28)\n", rows_as_int,columns_as_int);
    }

    unsigned char pixels[INSIZE][INSIZE];
    char label;

    for(int k = 0;k<10000;k++){
        
      //read current necessary data point
      fread(pixels,sizeof(pixels),1,images);
      fread(&label,sizeof(char),1,labels);


      //fill mnist_data struct -> data array with double values of pixels using cuda    
      unsigned char (*dpixels)[INSIZE];
      double (*ddata)[INSIZE];

      cudaMalloc((void**)&dpixels, INSIZE*INSIZE*sizeof(char));
      cudaMalloc((void**)&ddata, INSIZE*INSIZE*sizeof(double));

      cudaMemcpy(dpixels, pixels, INSIZE*INSIZE*sizeof(unsigned char), cudaMemcpyHostToDevice);

      dim3 blocks(1,1);
      dim3 threads(INSIZE,INSIZE);
      fillArr<<<blocks, threads>>>(dpixels,ddata);

      cudaMemcpy((*data_set+*count)->data, ddata, INSIZE*INSIZE*sizeof(double), cudaMemcpyDeviceToHost);

      cudaFree(dpixels); 
      cudaFree(ddata);

      //assign mnist_data struct -> label with label 
      (*data_set+*count)->label = (int)label;

      //increment count
      *count+=1;
    }

    //close files
    fclose(images);
    fclose(labels);

    return 0;
}

//Convolution layer. Filtering.
__global__ void conv_filtering(float d_data[28][28],
                    float d_weight[6][5][5],
                    float d_filter_output[6][24][24]){
        
  int local_row = threadIdx.y;
  int local_column = threadIdx.z;
  int feature = threadIdx.x;
  int global_row = blockIdx.x+threadIdx.y;
  int global_column = blockIdx.y+threadIdx.z;
  int output_row = blockIdx.x;
  int output_column = blockIdx.y;
  __shared__ float temp[FEATURES][CONV_FILTER][CONV_FILTER];
  __shared__ float pre_sum[FEATURES][CONV_FILTER];
  temp[feature][local_row][local_column] = d_data[global_row][global_column]*d_weight[feature][local_row][local_column];
  __syncthreads();
  if(local_column==0){
      float temp_sum = 0.0f;
      for(int i =0; i< CONV_FILTER;i++){
       temp_sum+=temp[feature][local_row][i];
      }
      pre_sum[feature][local_row] = temp_sum;
      __syncthreads();
      if(local_row==0){
        float sum = 0.0f;
        for(int i =0; i< CONV_FILTER;i++){
          sum+=pre_sum[feature][i];
        }
        d_filter_output[feature][output_row][output_column] = sum;
      }
  }
}

//Convolution layer. Biasing.
__global__ void conv_biasing(float d_filter_output[6][24][24],
                    float d_bias[6],
                    float d_bias_output[6][24][24]){
                        
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    int feature = blockIdx.z;

    d_bias_output[feature][x][y] = d_filter_output[feature][x][y] + d_bias[feature];
}

//Convolution layer. Sigmoid.
__global__ void conv_sigmoid(float d_bias_output[6][24][24],
                    float d_sigmoid_output[6][24][24]){
                        
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    int feature = blockIdx.z;

    d_sigmoid_output[feature][x][y] = 1/(1+expf((-1)*d_bias_output[feature][x][y]));
}

//SubSampling layer. Filtering.
__global__ void ss_filtering(float d_conv_output[6][24][24],
                    float d_weight[4][4],
                    float d_filter_output[6][6][6]){
        
  int local_row = threadIdx.y;
  int local_column = threadIdx.z;
  int feature = threadIdx.x;
  int global_row = blockIdx.x*blockDim.y+threadIdx.y;
  int global_column = blockIdx.y*blockDim.z+threadIdx.z;
  int output_row = blockIdx.x;
  int output_column = blockIdx.y;
  __shared__ float temp[FEATURES][SS_FILTER][SS_FILTER];
  temp[feature][local_row][local_column] = d_conv_output[feature][global_row][global_column]*d_weight[local_row][local_column];
  __syncthreads();
  if(local_row==0 && local_column==0){
      float sum = 0.0f;
      for(int i = 0; i<SS_FILTER; i++){
          for(int j =0; j<SS_FILTER; j++){
              sum+=temp[feature][i][j];
          }
      }
      d_filter_output[feature][output_row][output_column] = sum;
  }
}

//SubSampling layer. Biasing.
__global__ void ss_biasing(float d_filter_output[6][6][6],
                    float d_bias[1],
                    float d_bias_output[6][6][6]){
                        
    int x = threadIdx.x;
    int y = threadIdx.y;
    int feature = blockIdx.x;

    d_bias_output[feature][x][y] = d_filter_output[feature][x][y] + d_bias[0];
}

//SubSampling layer. Sigmoid.
__global__ void ss_sigmoid(float d_bias_output[6][6][6],
                    float d_sigmoid_output[6][6][6]){
                        
    int x = threadIdx.x;
    int y = threadIdx.y;
    int feature = blockIdx.x;

    d_sigmoid_output[feature][x][y] = 1/(1+expf((-1)*d_bias_output[feature][x][y]));
}

__global__ void fc_linear(float d_ss_output[6][6][6], float d_weight[10][6][6][6],float d_linear_output[10]){
    const int neuron = blockIdx.x;
    const int depth = blockIdx.y*blockDim.x+threadIdx.x;
    const int local_depth = threadIdx.x;
    const int row = threadIdx.y;
    const int column = threadIdx.z;
    __shared__ float temp[3][6][6];
    __shared__ float temp_sums[3][6];
    __shared__ float pre_sums[3];
    temp[local_depth][row][column] = d_ss_output[depth][row][column]*d_weight[neuron][depth][row][column];
    __syncthreads();
    if(column==0){
      float temp_sum = 0.0f;
      for(int i = 0; i<6;i++){
        temp_sum+=temp[local_depth][row][i];  
      }
      temp_sums[local_depth][row] = temp_sum;
      if(row==0){
        float pre_sum = 0.0f;
        for(int i = 0; i<6;i++){
          pre_sum+=temp_sums[local_depth][i];  
        }
        pre_sums[local_depth] = pre_sum;
        if(local_depth==0){
          float sum = 0.0f;
          for(int i = 0; i<3;i++){
            sum+=pre_sums[i];  
          }
          atomicAdd(&d_linear_output[neuron],sum);
        }
      }
    }
}

//Fully-connected layer.Biasing.
__global__ void fc_biasing(float d_linear_output[10], float d_bias[10],float d_bias_output[10]){
    const int idx = threadIdx.x;
    d_bias_output[idx] = d_linear_output[idx]+d_bias[idx];
}

//Fully-connected layer.Sigmoid.
__global__ void fc_sigmoid(float d_bias_output[10], float d_final_output[10]){
    const int idx = threadIdx.x;
    d_final_output[idx] = 1/(1+expf((-1)*d_bias_output[idx]));
}

class Conv{
  public:
    int filter_size, features_num, output_dim;
    float *weight, *bias,*filter_output, *bias_output, *final_output;
    Conv(int filter_size, int features_num, int output);
    void forward_pass(float data[IMAGE_WIDTH][IMAGE_HEIGHT]);
    void get_filter_output(float res[FEATURES][CONV_OUTPUT][CONV_OUTPUT]);
    void get_bias_output(float res[FEATURES][CONV_OUTPUT][CONV_OUTPUT]);
    void get_final_output(float res[FEATURES][CONV_OUTPUT][CONV_OUTPUT]);
    ~Conv();

};

Conv::Conv(int filter_size, int features_num, int output_dim){
    
    //Assigning attributes
    this->filter_size = filter_size;
    this->features_num = features_num;
    this->output_dim = output_dim;
    
    float w[features_num][filter_size][filter_size],b[features_num];

    //Assigning all values of 'weight' and 'bias' to -1.0f 
    for(int i = 0; i < features_num; i++){
      b[i] = INITIAL_WEIGHT_VALUE;
      for(int j = 0; j < filter_size; j++){
        for(int k = 0; k < filter_size; k++){
          w[i][j][k] = INITIAL_WEIGHT_VALUE;   
        }
      }
    }

    //CUDA memory allocation
    cudaMalloc((void **)&weight, features_num*filter_size*filter_size*sizeof(float));
    cudaMemcpy(weight, w, features_num*filter_size*filter_size*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&bias, features_num*sizeof(float));
    cudaMemcpy(bias, b, features_num*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&filter_output, features_num*output_dim*output_dim*sizeof(float));
    cudaMalloc((void **)&bias_output, features_num*output_dim*output_dim*sizeof(float));
    cudaMalloc((void **)&final_output, features_num*output_dim*output_dim*sizeof(float));
}

Conv::~Conv(){
    
  //CUDA memory deallocation  
  cudaFree(weight);
  cudaFree(bias);
  cudaFree(filter_output);
  cudaFree(bias_output);
  cudaFree(final_output);

}

class SS{
  public:
    int filter_size, features_num, output_dim;
    float *weight, *bias,*filter_output, *bias_output, *final_output;
    SS(int filter_size, int features_num, int output);
    void forward_pass(float conv_output[FEATURES][CONV_OUTPUT][CONV_OUTPUT]);
    void get_filter_output(float res[FEATURES][SS_OUTPUT][SS_OUTPUT]);
    void get_bias_output(float res[FEATURES][SS_OUTPUT][SS_OUTPUT]);
    void get_final_output(float res[FEATURES][SS_OUTPUT][SS_OUTPUT]);
    ~SS();

};

SS::SS(int filter_size, int features_num, int output_dim){
    
    //Assigning attributes
    this->filter_size = filter_size;
    this->features_num = features_num;
    this->output_dim = output_dim;
    
    float w[filter_size][filter_size],b[1];

    //Assigning all values of 'weight' and 'bias' to -1.0f 
    b[0] = INITIAL_WEIGHT_VALUE;
    for(int i = 0; i < filter_size; i++){
      for(int j= 0; j < filter_size; j++){
        w[i][j] = INITIAL_WEIGHT_VALUE;   
        }
    }
    

    //CUDA memory allocation
    cudaMalloc((void **)&weight, filter_size*filter_size*sizeof(float));
    cudaMemcpy(weight, w, filter_size*filter_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&bias, filter_size*filter_size*sizeof(float));
    cudaMemcpy(bias, b, sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&filter_output, features_num*output_dim*output_dim*sizeof(float));
    cudaMalloc((void **)&bias_output, features_num*output_dim*output_dim*sizeof(float));
    cudaMalloc((void **)&final_output, features_num*output_dim*output_dim*sizeof(float));
}

SS::~SS(){
    
  //CUDA memory deallocation  
  cudaFree(weight);
  cudaFree(bias);
  cudaFree(filter_output);
  cudaFree(bias_output);
  cudaFree(final_output);

}

class FC{
  public:
    int  neurons, output_dim;
    float *weight, *bias,*linear_output, *bias_output, *final_output;
    FC(int neurons, int output);
    void forward_pass(float ss_output[FEATURES][SS_OUTPUT][SS_OUTPUT]);
    void get_linear_output(float res[FC_OUTPUT]);
    void get_bias_output(float res[FC_OUTPUT]);
    void get_final_output(float res[FC_OUTPUT]);
    ~FC();

};

FC::FC(int neurons, int output_dim){
    
    //Assigning attributes
    this->neurons = neurons;
    this->output_dim = output_dim;
    
    float w[neurons][FEATURES][SS_OUTPUT][SS_OUTPUT],b[neurons],l_o[output_dim];

    //Assigning all values of 'weight' and 'bias' to 1.0f 
    for(int i = 0; i < neurons; i++){
      b[i] = INITIAL_FC_WEIGHT_VALUE;
      l_o[i] = 0.0f;
      for(int j= 0; j < FEATURES; j++){
          for(int k= 0; k < SS_OUTPUT; k++){
              for(int l= 0; l < SS_OUTPUT; l++){
                w[i][j][k][l] = INITIAL_FC_WEIGHT_VALUE; 
              }
          }  
        }
    }
    //CUDA memory allocation
    cudaMalloc((void **)&weight, neurons*FEATURES*SS_OUTPUT*SS_OUTPUT*sizeof(float));
    cudaMemcpy(weight, w, neurons*FEATURES*SS_OUTPUT*SS_OUTPUT*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&bias, neurons*sizeof(float));
    cudaMemcpy(bias, b, neurons*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&linear_output, output_dim*sizeof(float));
    cudaMemcpy(linear_output, l_o, output_dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&bias_output, output_dim*sizeof(float));
    cudaMalloc((void **)&final_output, output_dim*sizeof(float));
}

FC::~FC(){
    
  //CUDA memory deallocation  
  cudaFree(weight);
  cudaFree(bias);
  cudaFree(linear_output);
  cudaFree(bias_output);
  cudaFree(final_output);
}

//MaxError checker for convolution layer, prints "correct results" if everything is OK
void conv_maxError(float arr[FEATURES][CONV_OUTPUT][CONV_OUTPUT],const char type[]){
  float maxError = 0.0f;
  int i,j,k,stop = 0;
  float checker;
  if(strcmp(type,"conv_filtering") == 0){
    checker = 25.0f;
  }
  else if(strcmp(type,"conv_biasing") == 0){
    checker = 24.0f; 
  }
  else if(strcmp(type,"conv_sigmoid") == 0){
    checker = 1/(1+expf((-1)*(24))); 
  }
  else{
    printf("Problems with array type to check.");
    exit(1);
  }
  for(i=0; i<FEATURES; i++){
    for(j=0; j<CONV_OUTPUT; j++){
      for(k=0; k<CONV_OUTPUT; k++){
        maxError = max(maxError, abs(arr[i][j][k] - checker));
        if(maxError!=0.0f){
          printf("Incorrect %s results\n", type);
          stop = 1;
          break;
          }
      }
      if(stop==1) break;
    }
    if(stop==1) break;
  }
  if(i==FEATURES&&j==CONV_OUTPUT&&k==CONV_OUTPUT)
    printf("Correct %s results\n", type); 
}

//MaxError checker for subsampling layer, prints "correct results" if everything is OK
void ss_maxError(float arr[FEATURES][SS_OUTPUT][SS_OUTPUT],const char type[]){
  float maxError = 0.0f;
  int i,j,k,stop = 0;
  float checker;
  if(strcmp(type,"ss_filtering") == 0){
    checker = -16.0f; 
  }
  else if(strcmp(type,"ss_biasing") == 0){
    checker = -17.0f; 
  }
  else if(strcmp(type,"ss_sigmoid") == 0){
    checker = 1/(1+expf((-1)*(-17))); 
  }
  else{
    printf("Problems with array type to check.");
    exit(1);
  }
  for(i=0; i<FEATURES; i++){
    for(j=0; j<SS_OUTPUT; j++){
      for(k=0; k<SS_OUTPUT; k++){
        maxError = max(maxError,abs(arr[i][j][k] - checker));
        if(maxError!=0.0f){
          printf("Incorrect %s results\n", type);
          stop = 1;
          break;
        }      
      }
      if(stop==1) break;
    }
    if(stop==1) break;
  }
  if(i==FEATURES&&j==SS_OUTPUT&&k==SS_OUTPUT)
    printf("Correct %s results\n", type); 
}

//MaxError checker for fully-connected layer, prints "correct results" if everything is OK
void fc_maxError(float arr[NEURONS],const char type[]){
  float maxError = 0.0f;
  int i = 0;
  float checker;
  if(strcmp(type,"fc_linear") == 0){
    checker = 216.0f;
  }
  else if(strcmp(type,"fc_biasing") == 0){
    checker = 217.0f; 
  }
  else if(strcmp(type,"fc_sigmoid") == 0){
    checker = 1/(1+expf((-1)*(217))); 
  }
  else{
    printf("Problems with array type to check.");
    exit(1);
  }
  for(i=0; i<NEURONS; i++){
        maxError = max(maxError, abs(arr[i] - checker));
        if(maxError!=0.0f){
          printf("Incorrect %s results\n", type);
          break;       
        }
  }
  if(i==NEURONS)
    printf("Correct %s results\n", type); 
}

//Forward pass 
void forward_pass(float data[IMAGE_WIDTH][IMAGE_HEIGHT]){
    
    Conv conv = Conv(CONV_FILTER, FEATURES, CONV_OUTPUT);
    SS ss = SS(SS_FILTER, FEATURES, SS_OUTPUT);
    FC fc = FC(NEURONS, FC_OUTPUT);
    
    float (*kernel_data)[IMAGE_HEIGHT];

    cudaMalloc((void**)&kernel_data,IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(float));
    cudaMemcpy(kernel_data, data, IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(float), cudaMemcpyHostToDevice);

    dim3 conv_filter_blocks(CONV_OUTPUT, CONV_OUTPUT);
    dim3 conv_filter_thread(FEATURES, CONV_FILTER, CONV_FILTER);
    conv_filtering<<<conv_filter_blocks, conv_filter_thread>>>(kernel_data,
                                         (float (*)[CONV_FILTER][CONV_FILTER])conv.weight,
                                         (float (*)[CONV_OUTPUT][CONV_OUTPUT])conv.filter_output);
    cudaError_t conv_filter_checker = cudaGetLastError();
    if (conv_filter_checker!=cudaSuccess){
      printf("CONV FILTERING PROBLEM:: %s", cudaGetErrorString(conv_filter_checker));
      exit(1);
    }
    
    int conv_block_dim = CONV_OUTPUT/3;
    dim3 conv_bias_blocks(CONV_OUTPUT/conv_block_dim,CONV_OUTPUT/conv_block_dim,FEATURES);
    dim3 conv_bias_thread(conv_block_dim,conv_block_dim);
    conv_biasing<<<conv_bias_blocks, conv_bias_thread>>>((float (*)[CONV_OUTPUT][CONV_OUTPUT])conv.filter_output, 
                                                         conv.bias,
                                                        (float (*)[CONV_OUTPUT][CONV_OUTPUT])conv.bias_output);
    cudaError_t conv_bias_checker = cudaGetLastError();
    if (conv_bias_checker!=cudaSuccess){
      printf("CONV BIASING PROBLEM:: %s", cudaGetErrorString(conv_bias_checker));
      exit(1);
    }
    
    dim3 conv_sigmoid_blocks(CONV_OUTPUT/conv_block_dim,CONV_OUTPUT/conv_block_dim,FEATURES);
    dim3 conv_sigmoid_thread(conv_block_dim,conv_block_dim);
    conv_sigmoid<<<conv_sigmoid_blocks, conv_sigmoid_thread>>>((float (*)[CONV_OUTPUT][CONV_OUTPUT])conv.bias_output,
                                                    (float (*)[CONV_OUTPUT][CONV_OUTPUT])conv.final_output);
    cudaError_t conv_sigmoid_checker = cudaGetLastError();
    if (conv_sigmoid_checker!=cudaSuccess){
      printf("CONV SIGMOID PROBLEM:: %s", cudaGetErrorString(conv_sigmoid_checker));
      exit(1);
    }

    dim3 ss_filter_blocks(SS_OUTPUT, SS_OUTPUT);
    dim3 ss_filter_thread(FEATURES, SS_FILTER, SS_FILTER);
    ss_filtering<<<ss_filter_blocks, ss_filter_thread>>>((float (*)[24][24])conv.final_output,
                                         (float (*)[SS_FILTER])ss.weight,
                                         (float (*)[SS_OUTPUT][SS_OUTPUT])ss.filter_output);
    cudaError_t ss_filter_checker = cudaGetLastError();
    if (ss_filter_checker!=cudaSuccess){
      printf("SS FILTERING PROBLEM:: %s", cudaGetErrorString(ss_filter_checker));
      exit(1);
    }
      
    dim3 ss_bias_blocks(FEATURES);
    dim3 ss_bias_thread(SS_OUTPUT,SS_OUTPUT);
    ss_biasing<<<ss_bias_blocks, ss_bias_thread>>>((float (*)[SS_OUTPUT][SS_OUTPUT])ss.filter_output, (float (*))ss.bias,
                                                (float (*)[SS_OUTPUT][SS_OUTPUT])ss.bias_output);
    cudaError_t ss_bias_checker = cudaGetLastError();
    if (ss_bias_checker!=cudaSuccess){
      printf("SS BIASING PROBLEM:: %s", cudaGetErrorString(ss_bias_checker));
      exit(1);
    }
    
    dim3 ss_sigmoid_blocks(FEATURES);
    dim3 ss_sigmoid_thread(SS_OUTPUT,SS_OUTPUT);
    ss_sigmoid<<<ss_sigmoid_blocks, ss_sigmoid_thread>>>((float (*)[SS_OUTPUT][SS_OUTPUT])ss.bias_output,
                                                    (float (*)[SS_OUTPUT][SS_OUTPUT])ss.final_output);
    cudaError_t ss_sigmoid_checker = cudaGetLastError();
    if (ss_sigmoid_checker!=cudaSuccess){
      printf("SS SIGMOID PROBLEM:: %s", cudaGetErrorString(ss_sigmoid_checker));
      exit(1);
    }

    float fc_data[6][6][6];
    for(int i = 0; i<6;i++){
        for(int j = 0; j<6;j++){
            for(int k = 0; k<6;k++){
                fc_data[i][j][k] = 1.0f;
            }
        }
    }
    float (*d_fc_data)[6][6];
    cudaMalloc((void**)&d_fc_data,6*6*6*sizeof(float));
    cudaMemcpy(d_fc_data, fc_data, 6*6*6*sizeof(float), cudaMemcpyHostToDevice);

    int div = FEATURES/2;
    dim3 fc_linear_blocks(FC_OUTPUT, FEATURES/div);
    dim3 fc_linear_thread(div, SS_OUTPUT, SS_OUTPUT);
    fc_linear<<<fc_linear_blocks, fc_linear_thread>>>(d_fc_data,
                                         (float (*)[FEATURES][SS_OUTPUT][SS_OUTPUT])fc.weight,
                                         fc.linear_output);
    cudaError_t fc_linear_checker = cudaGetLastError();
    if (fc_linear_checker!=cudaSuccess){
      printf("FC LINEAR PROBLEM:: %s", cudaGetErrorString(fc_linear_checker));
      exit(1);
    }
      
    dim3 fc_bias_blocks(1);
    dim3 fc_bias_thread(NEURONS);
    fc_biasing<<<fc_bias_blocks, fc_bias_thread>>>(fc.linear_output, fc.bias,
                                            fc.bias_output);
    cudaError_t fc_bias_checker = cudaGetLastError();
    if (fc_bias_checker!=cudaSuccess){
      printf("FC BIASING PROBLEM:: %s", cudaGetErrorString(fc_bias_checker));
      exit(1);
    }
    
    dim3 fc_sigmoid_blocks(1);
    dim3 fc_sigmoid_thread(NEURONS);
    fc_sigmoid<<<fc_sigmoid_blocks, fc_sigmoid_thread>>>(fc.bias_output,
                                                  fc.final_output);
    cudaError_t fc_sigmoid_checker = cudaGetLastError();
    if (fc_sigmoid_checker!=cudaSuccess){
      printf("FC SIGMOID PROBLEM:: %s", cudaGetErrorString(fc_sigmoid_checker));
      exit(1);
    }
    
    cudaFree(kernel_data);

    float conv_filter_res[6][24][24],conv_bias_res[6][24][24],conv_final_res[6][24][24],
      ss_filter_res[6][6][6],ss_bias_res[6][6][6],ss_final_res[6][6][6],
      fc_linear_res[10],fc_bias_res[10],fc_final_res[10];

      cudaMemcpy(conv_filter_res, conv.filter_output, 6*24*24*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(conv_bias_res, conv.bias_output, 6*24*24*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(conv_final_res, conv.final_output, 6*24*24*sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(ss_filter_res, ss.filter_output, 6*6*6*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(ss_bias_res, ss.bias_output, 6*6*6*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(ss_final_res, ss.final_output, 6*6*6*sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(fc_linear_res, fc.linear_output, 10*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(fc_bias_res, fc.bias_output, 10*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(fc_final_res, fc.final_output, 10*sizeof(float), cudaMemcpyDeviceToHost);

      conv_maxError(conv_filter_res,"conv_filtering");
      conv_maxError(conv_bias_res,"conv_biasing");
      conv_maxError(conv_final_res,"conv_sigmoid");
      
      ss_maxError(ss_filter_res,"ss_filtering");
      ss_maxError(ss_bias_res,"ss_biasing");
      ss_maxError(ss_final_res,"ss_sigmoid");
      
      fc_maxError(fc_linear_res,"fc_linear");
      fc_maxError(fc_bias_res,"fc_biasing");
      fc_maxError(fc_final_res,"fc_sigmoid");
}

int main(){
    
    const char *image_filename = "data/t10k-images.idx3-ubyte";
    const char *label_filename = "data/t10k-labels.idx1-ubyte";
    mnist_data *data_set = (mnist_data *)malloc(sizeof(*data_set)*10000);
    unsigned int count = 0;
    
    if(mnist_load(image_filename,label_filename, &data_set,&count)!=0){
      printf("Problems with loading data.");
      exit(1);
    }
    printf("test_cnt = %d (should be 10000)\n\n",count);
    
    float data[IMAGE_HEIGHT][IMAGE_WIDTH];
    for(int i = 0; i< IMAGE_HEIGHT;i++){
        for(int j = 0; j< IMAGE_WIDTH;j++){
            data_set[0].data[i][j] = -1.0;
            data[i][j] = data_set[0].data[i][j];
        }
    }
    forward_pass(data);
    return 0;
}
