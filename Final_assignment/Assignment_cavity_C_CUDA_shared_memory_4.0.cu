#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<limits.h>
#include<algorithm>
#include<chrono>
#include<iostream>
#include"vtr_writer.hpp"

//nx=41,ny=41,nt=10000, my laptop 17.79MLUPS, my desktop 21.47 MLUPS,   g++ Assignment_cavity_C_2.0.cpp -O3 -march=native

const int nx = 40 + 1; 
const int ny = 40 + 1;
const int nt = 10000;
const int nit = 50;
const double dx = 2.0 / (nx-1);
const double dy = 2.0 / (ny-1);
const double dt = 0.01;
const double rho = 1.0;
const double nu = 0.02;

const int num_threads = 1024;
dim3 dimBlock( num_threads, 1);    //dimBlock = (1024,1)
dim3 grid( (nx + num_threads - 1) / num_threads, ny);  //grid = (1,41) if nx,ny=(41,41)

void initialize_variables(double *u, double *v, double *p, double *un, double *vn, double *pn, double *b);
__global__ void calculate_b(double *b, double *u, double *v);

void calculate_pressure(double *p, double *pn, double *b);
__global__ void update_pressure(double *p, double *pn);
__global__ void calculate_poisson_equation(double *p, double *pn, double *b);
__global__ void boundary_condition_pressure(double *p);

void calculate_velocity(double *u, double *v, double *un, double *vn, double *p);
__global__ void update_velocity(double *u, double *v, double *un, double *vn);
__global__ void calculate_advection_diffusion(double *u, double *v, double *un, double *vn, double *p);
__global__ void boundary_condition_velocity(double *u, double *v);
void output_vtr(int outputStep, double simTime, double *u, double *v, double *p);

int main(){
    int size = nx * ny * sizeof(double);
    double *u, *v, *p, *un, *vn, *pn, *b;
    cudaMallocManaged(&u,  size);
    cudaMallocManaged(&v,  size);
    cudaMallocManaged(&p,  size);
    cudaMallocManaged(&un, size);
    cudaMallocManaged(&vn, size);
    cudaMallocManaged(&pn, size);
    cudaMallocManaged(&b,  size);

    const double MLU = (nx-1) * (ny-1) * nt / 1.e+6; //Mega Lattice Update
    double MLUPS = 0.0; //Mega Lattice Update Per Second

    initialize_variables(u, v, p, un, vn, pn, b);  //0 initialize velocity and pressure
    //main roop start from here
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int n=0; n<nt; n++){
        calculate_b<<<grid, dimBlock>>>(b, u, v);
        cudaDeviceSynchronize();

        calculate_pressure(p, pn, b);
        calculate_velocity(u, v, un, vn, p);
        
        if (n % 1000 == 0){
            output_vtr(n, nt*dt, u, v, p);  //output visualize file of pressure
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Main roop calculate time: " << elapsed_time.count() << " seconds" << std::endl;
    MLUPS = MLU / elapsed_time.count();
    std::cout << MLUPS << "MLUPS" << std::endl;

    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(un);
    cudaFree(vn);
    cudaFree(pn);
    cudaFree(b);
    return 0;
}

void initialize_variables(double *u, double *v, double *p, double *un, double *vn, double *pn, double *b){
    int size = nx * ny * sizeof(double);
    if (u != NULL) cudaMemset(u,0,size);
    if (v != NULL) cudaMemset(v,0,size);
    if (p != NULL) cudaMemset(p,0,size);
    if (un != NULL) cudaMemset(un,0,size);
    if (vn != NULL) cudaMemset(vn,0,size);
    if (pn != NULL) cudaMemset(pn,0,size);
    if (b != NULL) cudaMemset(b,0,size);
}

__global__ void calculate_b(double *b, double *u, double *v){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    int index = idy*nx + idx; //index = [j][i], index+1 = [j][i+1], index+nx = [j+1][i]
    if (idx > 0 && idx < nx-1 && idy > 0 && idy < ny-1){
        b[index] = rho * (1.0 / dt *\
                        ((u[index+1] - u[index-1]) / (2 * dx) + (v[index+nx] - v[index-nx]) / (2 * dy)) -\
                   pow( ((u[index+1] - u[index-1]) / (2 * dx)), 2) - 2 * ((u[index+nx] - u[index-nx]) / (2 * dy) *\
                         (v[index+1] - v[index-1]) / (2 * dx)) - pow(((v[index+nx] - v[index-nx]) / (2 * dy)), 2) );
    }
}

void calculate_pressure(double *p, double *pn, double *b){
    for(int i = 0; i < nit; i++){
        update_pressure<<<grid, dimBlock>>>(p, pn);
        cudaDeviceSynchronize();
        calculate_poisson_equation<<<grid, dimBlock>>>(p, pn, b);
        cudaDeviceSynchronize();
        boundary_condition_pressure<<<grid, dimBlock>>>(p);
        cudaDeviceSynchronize();
    }
}

__global__ void update_pressure(double *p, double *pn){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    int index = idy*nx + idx; //index = [j][i], index+1 = [j][i+1], index+nx = [j+1][i]
    if (idx < nx && idy < ny){
        pn[index] = p[index];
    }
}

__global__ void calculate_poisson_equation(double *p, double *pn, double *b){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    int index = idy*nx + idx; //index = [j][i], index+1 = [j][i+1], index+nx = [j+1][i]
    if (idx >= 1 && idx < nx-1 && idy >= 1 && idy < ny-1){
        p[index] = (pow(dy,2) * (pn[index+1] + pn[index-1]) +\
                    pow(dx,2) * (pn[index+nx] + pn[index-nx]) -\
                    b[index] * pow(dx,2) * pow(dy,2) )\
                    / (2 * ( pow(dx,2) + pow(dy,2) ));
    }
}

__global__ void boundary_condition_pressure(double *p){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    int index = idy*nx + idx; //index = [j][i], index+1 = [j][i+1], index+nx = [j+1][i]
    if (idx == 0 && idy < ny){
        p[index] = p[index+1];      //left
    }
    if (idx == nx-1 && idy < ny){
        p[index] = p[index-1];      //right
    }
    if (idy ==0 && idx < nx){
        p[index] = p[index+nx];     //bottom
    }
    if (idy == ny-1 && idx < nx){
        p[index] = 0.0;            //top
    }
}

void calculate_velocity(double *u, double *v, double *un, double *vn, double *p){
    update_velocity<<<grid, dimBlock>>>(u, v, un, vn);
    cudaDeviceSynchronize();
    calculate_advection_diffusion<<<grid, dimBlock>>>(u, v, un, vn, p);
    cudaDeviceSynchronize();
    boundary_condition_velocity<<<grid, dimBlock>>>(u, v);
    cudaDeviceSynchronize();
}

__global__ void update_velocity(double *u, double *v, double *un, double *vn){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    int index = idy*nx + idx; //index = [j][i], index+1 = [j][i+1], index+nx = [j+1][i]
    if (idx < nx && idy < ny){
        un[index] = u[index];
        vn[index] = v[index];
    }
}

__global__ void calculate_advection_diffusion(double *u, double *v, double *un, double *vn, double *p){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    int index = idy*nx + idx; //index = [j][i], index+1 = [j][i+1], index+nx = [j+1][i]
    if (idx >= 1 && idx < nx-1 && idy >= 1 && idy < ny-1){
        u[index] = un[index] - un[index] * dt / dx * (un[index] - un[index-1])\
                             -un[index] * dt / dy * (un[index] - un[index-nx])\
                             -dt / (2 * rho * dx) * (p[index+1] - p[index-1])\
                             + nu * dt / pow(dx,2) * (un[index+1] - 2 * un[index] + un[index-1])\
                             + nu * dt / pow(dy,2) * (un[index+nx] - 2 * un[index] + un[index-nx]);
        v[index] = vn[index] - vn[index] * dt / dx * (vn[index] - vn[index-1])\
                             - vn[index] * dt / dy * (vn[index] - vn[index-nx])\
                             - dt / (2 * rho * dx) * (p[index+nx] - p[index-nx])\
                             + nu * dt / pow(dx,2) * (vn[index+1] - 2 * vn[index] + vn[index-1])\
                             + nu * dt / pow(dy,2) * (vn[index+nx] - 2 * vn[index] + vn[index-nx]);
    }
}

__global__ void boundary_condition_velocity(double *u, double *v){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    int index = idy*nx + idx; //index = [j][i], index+1 = [j][i+1], index+nx = [j+1][i]
    if (idy ==0 && idx < nx){
        u[index] = 0.0;       //bottom
        v[index] = 0.0;       //bottom
    }
    if (idy == ny-1 && idx < nx){
        u[index] = 1.0;       //top
        v[index] = 0.0;       //top
    }
    if (idx == 0 && idy < ny){
        u[index] = 0.0;       //left
        v[index] = 0.0;       //left
    }
    if (idx == nx-1 && idy < ny){
        u[index] = 0.0;       //right
        v[index] = 0.0;       //right
    }
}

void output_vtr(int outputStep, double simTime, double *u, double *v, double *p)
{
    // file name & dir path setting
    char dir_path[1024],output_dir[512],filename[256];
    sprintf(output_dir, ".");
    sprintf(filename, "CavityFlow");
    sprintf(dir_path, "%s/%s", output_dir, filename);

    // make buffers for cell centered data
    std::vector<double> buff_p;
    std::vector<double> buff_u, buff_v, buff_w;
    std::vector<double> x,y,z;


    const int ist = 0, ien = nx - 1 ;
    const int jst = 0, jen = ny - 1 ;

    // set coordinate
    for(int j=jst; j<=jen; j++){ y.push_back( (j - jst)*dy ); }
    for(int i=ist; i<=ien; i++){ x.push_back( (i - ist)*dx ); }

    z.push_back( 0.0 );

    for(int j = jst; j < jen; j++)
    for(int i = ist; i < ien; i++)
    {
        buff_u.push_back( u[i+j*nx] );
        buff_v.push_back( v[i+j*nx] );

        buff_w.push_back( 0.0 );

        buff_p.push_back( p[i+j*nx] );
    }
    flow::vtr_writer vtr;
    vtr.init(dir_path,filename, nx  , ny , 1 , 0, nx - 1 , 0, ny - 1, 0, 0, true);
    vtr.set_coordinate(&x.front(),&y.front(),&z.front());
    vtr.push_cell_array("velocity", &buff_u.front(), &buff_v.front(), &buff_w.front());
    vtr.push_cell_array("pressure", &buff_p.front());
    vtr.set_current_step(outputStep);
    vtr.write(simTime);
}