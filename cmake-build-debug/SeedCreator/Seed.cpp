#include <iostream>
#include <fstream>
#include "immintrin.h"

int main(int argc, char *argv[]){

  size_t size_file;
  size_file = std::strtoul(argv[1], nullptr, 0);

  std::ofstream seeds_file("seeds.dat");

  long long unsigned int seed;
  // Cuantas semillas queremos?
  for (size_t i = 0; i < size_file; i++) {
    while (!_rdseed64_step(&seed)){

    }

    seeds_file << seed << "\n";
  }

  seeds_file.close();

  return 0;
}