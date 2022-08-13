#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <curand_kernel.h>

__device__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z){
  return (x & y) ^ ((~x) & z);
}

__device__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z){
  return (x & y) ^ (x & z) ^ (y & z);
}

__device__ uint32_t ROTR(uint32_t x, unsigned long n){
  return (x >> n) | ( x << (32-n));
}

__device__ uint32_t SHR(uint32_t x, unsigned long n){
  return (x >> n);
}

__device__ uint32_t Sigma_0(uint32_t x){
  return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22);
}

__device__ uint32_t Sigma_1(uint32_t x){
  return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25);
}

__device__ uint32_t sigma_0(uint32_t x){
  return ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3);
}

__device__ uint32_t sigma_1(uint32_t x){
  return ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10);
}

__device__ __constant__ uint32_t constants[64] = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
                                                  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
                                                  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
                                                  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
                                                  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
                                                  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
                                                  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
                                                  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

__device__ __constant__ uint32_t padding[14] = {0x80000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,
                                                0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000040};

__device__ __constant__ uint32_t masks[8] = {0x0000000F,0x000000FF,0x00000FFF,0x0000FFFF,0x000FFFFF,0x00FFFFFF,0x0FFFFFFF,0xFFFFFFFF};

__global__ void setup_kernel(curandState *state, unsigned long long int seed)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets same seed (cryptographic quality), a different sequence
     number, no offset */
  curand_init(seed, id, 0, &state[id]);
}

// Sabemos que m tiene 2 indices y su padding ya está en el device
__global__ void SHA_256_64(curandState *state, uint32_t *hash, uint32_t *m_device) {

  // 64 bit message
  uint32_t m[2];
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Copy state to local memory for efficiency */
  curandState localState = state[id];
  /* Generate pseudo-random unsigned ints */
  uint32_t h[8];
  uint32_t W[64];
  m[0] = curand(&localState);
  m[1] = curand(&localState);

  m_device[id * 2] = m[0];
  m_device[id * 2 + 1] = m[1];

  h[0] = 0x6a09e667;
  h[1] = 0xbb67ae85;
  h[2] = 0x3c6ef372;
  h[3] = 0xa54ff53a;
  h[4] = 0x510e527f;
  h[5] = 0x9b05688c;
  h[6] = 0x1f83d9ab;
  h[7] = 0x5be0cd19;

  // Message schedule
  W[0] = m[0];
  W[1] = m[1];
  W[2] = padding[0];
  W[3] = padding[1];
  W[4] = padding[2];
  W[5] = padding[3];
  W[6] = padding[4];
  W[7] = padding[5];
  W[8] = padding[6];
  W[9] = padding[7];
  W[10] = padding[8];
  W[11] = padding[9];
  W[12] = padding[10];
  W[13] = padding[11];
  W[14] = padding[12];
  W[15] = padding[13];

  for(int i = 16; i < 64; ++i){
    W[i] = sigma_1(W[i-2]) + W[i-7] + sigma_0(W[i-15]) + W[i-16];
  }

  uint32_t a = h[0];
  uint32_t b = h[1];
  uint32_t c = h[2];
  uint32_t d = h[3];
  uint32_t e = h[4];
  uint32_t f = h[5];
  uint32_t g = h[6];
  uint32_t htmp = h[7];

  for(int i = 0; i < 64; ++i){
    uint32_t T_1 = htmp + Sigma_1(e) + Ch(e,f,g) + constants[i] + W[i];
    uint32_t T_2 = Sigma_0(a) + Maj(a,b,c);
    htmp = g;
    g = f;
    f = e;
    e = d + T_1;
    d = c;
    c = b;
    b = a;
    a = T_1 + T_2;
  }

  h[0] += a;
  h[1] += b;
  h[2] += c;
  h[3] += d;
  h[4] += e;
  h[5] += f;
  h[6] += g;
  h[7] += htmp;

  hash[id * 8] = h[0];
  hash[id * 8 + 1] = h[1];
  hash[id * 8 + 2] = h[2];
  hash[id * 8 + 3] = h[3];
  hash[id * 8 + 4] = h[4];
  hash[id * 8 + 5] = h[5];
  hash[id * 8 + 6] = h[6];
  hash[id * 8 + 7] = h[7];
}

__global__ void found(uint32_t *hash, bool *found, uint32_t number_of_zeroes){
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  uint32_t number_zerous_tmp = 0;

  if((hash[id * 8 + 7] & masks[0]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 7] & masks[1]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 7] & masks[2]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 7] & masks[3]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 7] & masks[4]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 7] & masks[5]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 7] & masks[6]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 7] & masks[7]) == 0)
    number_zerous_tmp++;

  if(number_of_zeroes == number_zerous_tmp){
    found[id] = true;
    return;
  } else if (number_of_zeroes < number_zerous_tmp){
    found[id] = false;
    return;
  } else if(number_zerous_tmp != 8){
    found[id] = false;
    return;
  }

  if((hash[id * 8 + 6] & masks[0]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 6] & masks[1]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 6] & masks[2]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 6] & masks[3]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 6] & masks[4]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 6] & masks[5]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 6] & masks[6]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 6] & masks[7]) == 0)
    number_zerous_tmp++;

  if(number_of_zeroes == number_zerous_tmp){
    found[id] = true;
    return;
  } else if (number_of_zeroes < number_zerous_tmp){
    found[id] = false;
    return;
  } else if(number_zerous_tmp != 16){
    found[id] = false;
    return;
  }

  if((hash[id * 8 + 5] & masks[0]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 5] & masks[1]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 5] & masks[2]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 5] & masks[3]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 5] & masks[4]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 5] & masks[5]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 5] & masks[6]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 5] & masks[7]) == 0)
    number_zerous_tmp++;

  if(number_of_zeroes == number_zerous_tmp){
    found[id] = true;
    return;
  } else if (number_of_zeroes < number_zerous_tmp){
    found[id] = false;
    return;
  } else if(number_zerous_tmp != 24){
    found[id] = false;
    return;
  }

  if((hash[id * 8 + 4] & masks[0]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 4] & masks[1]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 4] & masks[2]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 4] & masks[3]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 4] & masks[4]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 4] & masks[5]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 4] & masks[6]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 4] & masks[7]) == 0)
    number_zerous_tmp++;

  if(number_of_zeroes == number_zerous_tmp){
    found[id] = true;
    return;
  } else if (number_of_zeroes < number_zerous_tmp){
    found[id] = false;
    return;
  } else if(number_zerous_tmp != 32){
    found[id] = false;
    return;
  }

  if((hash[id * 8 + 3] & masks[0]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 3] & masks[1]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 3] & masks[2]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 3] & masks[3]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 3] & masks[4]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 3] & masks[5]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 3] & masks[6]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 3] & masks[7]) == 0)
    number_zerous_tmp++;

  if(number_of_zeroes == number_zerous_tmp){
    found[id] = true;
    return;
  } else if (number_of_zeroes < number_zerous_tmp){
    found[id] = false;
    return;
  } else if(number_zerous_tmp != 40){
    found[id] = false;
    return;
  }

  if((hash[id * 8 + 2] & masks[0]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 2] & masks[1]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 2] & masks[2]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 2] & masks[3]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 2] & masks[4]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 2] & masks[5]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 2] & masks[6]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 2] & masks[7]) == 0)
    number_zerous_tmp++;

  if(number_of_zeroes == number_zerous_tmp){
    found[id] = true;
    return;
  } else if (number_of_zeroes < number_zerous_tmp){
    found[id] = false;
    return;
  } else if(number_zerous_tmp != 48){
    found[id] = false;
    return;
  }

  if((hash[id * 8 + 1] & masks[0]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 1] & masks[1]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 1] & masks[2]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 1] & masks[3]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 1] & masks[4]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 1] & masks[5]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 1] & masks[6]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8 + 1] & masks[7]) == 0)
    number_zerous_tmp++;

  if(number_of_zeroes == number_zerous_tmp){
    found[id] = true;
    return;
  } else if (number_of_zeroes < number_zerous_tmp){
    found[id] = false;
    return;
  } else if(number_zerous_tmp != 56){
    found[id] = false;
    return;
  }

  if((hash[id * 8] & masks[0]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8] & masks[1]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8] & masks[2]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8] & masks[3]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8] & masks[4]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8] & masks[5]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8] & masks[6]) == 0)
    number_zerous_tmp++;

  if((hash[id * 8] & masks[7]) == 0)
    number_zerous_tmp++;

  if(number_of_zeroes == number_zerous_tmp){
    found[id] = true;
    return;
  } else {
    found[id] = false;
    return;
  }
}

int main() {
  const unsigned int threadsPerBlock = 16;
  const unsigned int blockCount = 15000;
  const unsigned int totalThreads = threadsPerBlock * blockCount;
  const uint32_t number_of_trailing_zeroes = 7;

  std::ifstream seeds("seeds.dat");
  unsigned long long int seed;
  bool found_flag = false;
  while(!seeds.eof()){
    seeds >> seed;
    curandState *devStates;
    cudaMalloc((void **)&devStates, totalThreads * sizeof(curandState));
    setup_kernel<<<blockCount, threadsPerBlock>>>(devStates, seed);

    bool *found_host = new bool[totalThreads]();
    bool *found_device;
    uint32_t *m_host = new uint32_t[totalThreads * 2];
    uint32_t *m_device;
    uint32_t *hash_host = new uint32_t[totalThreads * 8];
    uint32_t *hash_device;
    cudaMalloc(&hash_device, totalThreads * 8 * sizeof(uint32_t));
    cudaMalloc(&m_device, totalThreads * 2 * sizeof(uint32_t));
    cudaMalloc(&found_device, totalThreads * sizeof(bool));
    cudaMemcpy(hash_device, hash_host, totalThreads * 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(m_device, m_host, totalThreads * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(found_device, found_host, totalThreads * sizeof(bool), cudaMemcpyHostToDevice);
    SHA_256_64<<<blockCount, threadsPerBlock>>>(devStates, hash_device, m_device);
    found<<<blockCount, threadsPerBlock>>>(hash_device, found_device, number_of_trailing_zeroes);
    cudaDeviceSynchronize();
    cudaMemcpy(hash_host, hash_device, totalThreads * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(found_host, found_device, totalThreads * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_host, m_device, totalThreads * 2 * sizeof(bool), cudaMemcpyDeviceToHost);

    for(size_t index_hashes = 0; index_hashes < totalThreads; ++index_hashes){
      if(found_host[index_hashes]){
        found_flag = true;
        std::cout << "Message: " << std::hex << std::setw(8) << std::setfill('0') << m_host[index_hashes*2] << " ";
        std::cout << std::hex << std::setw(8) << std::setfill('0') << m_host[index_hashes*2 + 1]<< "\n";
        std::cout << "Hash: ";
        for(size_t index_in_hash = 0; index_in_hash < 8; ++index_in_hash){
          std::cout << std::hex << std::setw(8) << std::setfill('0') << std::hex << hash_host[index_hashes * 8 + index_in_hash] << " ";
        }
        std::cout << "\n";
        break;
      }
    }

    cudaFree(hash_device);
    cudaFree(found_device);
    cudaFree(m_device);
    free(m_host);
    free(found_host);
    free(hash_host);
    if(found_flag)
      break;
  }
  return 0;
}
