__kernel void get_gray_binary(__global float *img,
 __global float *result,__global int *t1, __global int *t2,
  __global int *W){
    int w = *W;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w*posy + posx;
    if(img[i] > *t1 && img[i] < *t2)
        result[i] = 1;
    else
        result[i] =0;
}



