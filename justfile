alias bc := build-cuda
alias bm := build-mpi
up:
  docker compose up -d --build
down:
  docker compose down

build-cuda:
  docker compose exec -T cuda-dev nvcc -c src/MPI-CUDA.cu -o build/MPI-CUDA.o

build-mpi:
  docker compose exec -T cuda-dev mpicc -o build/MPI-CUDA.exe build/MPI-CUDA.o -L/usr/local/cuda/lib64/ -lcudart -lstdc++

build: build-cuda build-mpi

clean:
  docker compose exec -T cuda-dev rm -f build/MPI-CUDA.o build/MPI-CUDA.exe

run n="2" k="1024" mode="V" data="data.txt":
  docker compose exec -T cuda-dev bash -c "mpirun -np {{n}} ./build/MPI-CUDA.exe {{k}} -{{mode}} < {{data}}"

