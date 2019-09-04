__constant int sobel9[9][9] = {{-16575,  -15912,  -13260,   -7800 ,      0  ,  7800 ,  13260 ,  15912 ,  16575},
{-21216 , -22100 , -20400 , -13260 ,      0  , 13260 ,  20400  , 22100 ,  21216},
{-26520 , -30600 , -33150 , -26520    ,   0  , 26520  , 33150,   30600,   26520},
{-31200  ,-39780 , -53040 , -66300  ,     0  , 66300 ,  53040 ,  39780 ,  31200},
{ -33150 , -44200 , -66300, -132600     ,  0 , 132600 ,  66300 ,  44200 ,  33150},
{-31200 , -39780 , -53040 , -66300   ,    0  , 66300  , 53040  , 39780 ,  31200},
{-26520 , -30600 , -33150 , -26520  ,     0  , 26520 ,  33150 ,  30600 ,  26520},
{-21216 , -22100  ,-20400 , -13260   ,    0  , 13260  , 20400  , 22100 ,  21216},
{-16575 , -15912 , -13260 ,  -7800   ,    0   , 7800 ,  13260  , 15912 ,  16575}
};




__kernel void sobelXFilter(__global float *img, __global float *result, __global int *width, __global int *height){
    int w = *width;
    int h = *height;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w*posy + posx;
    
    if(posx == 0 || posy ==0 || posx==w-1 || posy == h-1){
         result[i] = img[i] ;  
    }else{
        int pixel00, pixel02, pixel10, pixel12, pixel20, pixel22;
        pixel00 = -1*img[i - w-1];
        pixel02 =  img[i - w+1];
        pixel10 = -2*img[i    -1];
        pixel12 =  2*img[i    +1];
        pixel20 = -1*img[i + w-1];
        pixel22 =  img[i + w+1];
        result[i] = pixel00+pixel02+pixel10+pixel12+pixel20+pixel22;
    }
}
__kernel void sobelYFilter(__global float *img, __global float *result, __global int *width, __global int *height){
    int w = *width;
    int h = *height;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w*posy + posx;
    
    if(posx == 0 || posy ==0 || posx==w-1 || posy == h-1){
         result[i] = img[i] ;  
    }else{
        int pixel00, pixel01, pixel02, pixel20, pixel21, pixel22;
        pixel00 =    img[i-w -1 ];
        pixel01 =  2*img[i -w ];
        pixel02 =    img[i  -w + 1 ];
        pixel20 =  -1*img[i +w-1 ];
        pixel21 =   -2*img[i+w ];
        pixel22 =  -1*img[i+w+1 ];
        result[i] = pixel00+pixel01+pixel02+pixel20+pixel21+pixel22;
    }
}
__kernel void sobelXFilter9x9(__global float *img, __global float *result, __global int *width, __global int *height){
    int w = *width;
    int h = *height;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w*posy + posx;
    int inc;
    long pixels[9] = {0,0,0,0,0,0,0,0,0};   
    if(posx <= 4 || posy <=4 || posx>=w-4 || posy >= h-4){
         result[i] = img[i] ;  
    }else{
        for(inc = -4; inc < 5; ++inc){
            pixels[0] += sobel9[inc+4][0]*img[i+inc*w-4];
            pixels[1] += sobel9[inc+4][1]*img[i+inc*w-3];
            pixels[2] += sobel9[inc+4][2]*img[i+inc*w-2];
            pixels[3] += sobel9[inc+4][3]*img[i+inc*w-1];
            pixels[4] += sobel9[inc+4][4]*img[i+inc*w];
            pixels[5] += sobel9[inc+4][5]*img[i+inc*w+1];
            pixels[6] += sobel9[inc+4][6]*img[i+inc*w+2];
            pixels[7] += sobel9[inc+4][7]*img[i+inc*w+3];
            pixels[8] += sobel9[inc+4][8]*img[i+inc*w+4];
        }
           
        result[i] = pixels[0]+pixels[1]+pixels[2]+pixels[3]+pixels[4]+pixels[5]+pixels[6]+pixels[7]+pixels[8];
    }
}
__kernel void sobelYFilter9x9(__global float *img, __global float *result, __global int *width, __global int *height){
    int w = *width;
    int h = *height;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w*posy + posx;
    int inc;
    int k =8;
    long pixels[9] = {0,0,0,0,0,0,0,0,0};   
    if(posx <= 4 || posy <=4 || posx>=w-4 || posy >= h-4){
         result[i] = img[i] ;  
    }else{
            for(inc = -4; inc < 5; ++inc){
            pixels[0] += sobel9[0][k]*img[i+inc*w-4];
            pixels[1] += sobel9[1][k]*img[i+inc*w-3];
            pixels[2] += sobel9[2][k]*img[i+inc*w-2];
            pixels[3] += sobel9[3][k]*img[i+inc*w-1];
            pixels[4] += sobel9[4][k]*img[i+inc*w];
            pixels[5] += sobel9[5][k]*img[i+inc*w+1];
            pixels[6] += sobel9[6][k]*img[i+inc*w+2];
            pixels[7] += sobel9[7][k]*img[i+inc*w+3];
            pixels[8] += sobel9[8][k]*img[i+inc*w+4];
            --k;
        }
           
        result[i] = pixels[0]+pixels[1]+pixels[2]+pixels[3]+pixels[4]+pixels[5]+pixels[6]+pixels[7]+pixels[8];
    }
}



__kernel void magnitude(__global float *img1,__global float *img2, __global float *result,__global int *width){
    int w = *width;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w*posy + posx;
    
    result[i] = sqrt(img1[i]*img1[i] + img2[i]*img2[i]);
    
}

