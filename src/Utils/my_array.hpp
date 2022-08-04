// Copyright (c) "2019, by Stanford University
//               Developer: Mario Di Renzo
//               Affiliation: Center for Turbulence Research, Stanford University
//               URL: https://ctr.stanford.edu
//               Citation: Di Renzo, M., Lin, F., and Urzay, J. (2020).
//                         HTR solver: An open-source exascale-oriented task-based
//                         multi-GPU high-order code for hypersonic aerothermodynamics.
//                         Computer Physics Communications 255, 107262"
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef __MY_ARRAY_HPP__
#define __MY_ARRAY_HPP__

#ifndef __CUDA_HD__
#ifdef __CUDACC__
#define __CUDA_HD__ __host__ __device__
#else
#define __CUDA_HD__
#endif
#endif

#ifndef __CUDA_H__
#ifdef __CUDACC__
#define __CUDA_H__ __device__
#else
#define __CUDA_H__
#endif
#endif

#ifndef __UNROLL__
#ifdef __CUDACC__
#define __UNROLL__ #pragma unroll
#else
#define __UNROLL__
#endif
#endif

template<typename T, int SIZE>
struct MyArray {
public:
   __CUDA_HD__
   inline MyArray() {}

   // BE CAREFULL THIS DOES NOT CHEK THE SANITY OF THE INPUT
   __CUDA_HD__
   inline MyArray(const T *in) {
      __UNROLL__
      for (int i=0; i<SIZE; i++)
            v[i] = in[i];
   }

   __CUDA_HD__
   inline void init(const T in) {
      __UNROLL__
      for (int i=0; i<SIZE; i++)
         (*this)[i] = in;
   }

   __CUDA_HD__
   inline T& operator[](const int index)       {
#ifdef LEGION_BOUNDS_CHECKS
      assert(index >=   0);
      assert(index < SIZE);
#endif
      return v[index];
   }

   __CUDA_HD__
   inline T  operator[](const int index) const {
#ifdef LEGION_BOUNDS_CHECKS
      assert(index >=   0);
      assert(index < SIZE);
#endif
      return v[index];
   }

   __CUDA_HD__
   inline MyArray<T, SIZE>& operator=(const MyArray<T, SIZE> &in) {
      __UNROLL__
      for (int i=0; i<SIZE; i++)
         (*this)[i] = in[i];
      return *this;
   }

   __CUDA_HD__
   inline MyArray<T, SIZE>& operator+=(const MyArray<T, SIZE> &rhs) {
      __UNROLL__
      for (int i=0; i<SIZE; i++)
         (*this)[i] += rhs[i];
      return *this;
   }

   __CUDA_HD__
   friend inline MyArray<T, SIZE> operator+(MyArray<T, SIZE> lhs, const  MyArray<T, SIZE> &rhs) {
      lhs += rhs;
      return lhs;
   }

   __CUDA_HD__
   inline MyArray<T, SIZE>& operator-=(const MyArray<T, SIZE> &rhs) {
      __UNROLL__
      for (int i=0; i<SIZE; i++)
         (*this)[i] -= rhs[i];
      return *this;
   }

   __CUDA_HD__
   friend inline MyArray<T, SIZE> operator-(MyArray<T, SIZE> lhs, const  MyArray<T, SIZE> &rhs) {
      lhs -= rhs;
      return lhs;
   }

   __CUDA_HD__
   inline MyArray<T, SIZE>& operator-() {
      __UNROLL__
      for (int i=0; i<SIZE; i++)
         (*this)[i] = -(*this)[i];
      return *this;
   }

   __CUDA_HD__
   inline MyArray<T, SIZE>& operator+=(const T rhs) {
      __UNROLL__
      for (int i=0; i<SIZE; i++)
         (*this)[i] += rhs;
      return *this;
   }

   __CUDA_HD__
   inline MyArray<T, SIZE>& operator*=(const T rhs) {
      __UNROLL__
      for (int i=0; i<SIZE; i++)
         (*this)[i] *= rhs;
      return *this;
   }

   __CUDA_HD__
   friend inline MyArray<T, SIZE> operator+(MyArray<T, SIZE> lhs, const T rhs) {
      lhs += rhs;
      return lhs;
   }

   __CUDA_HD__
   friend inline MyArray<T, SIZE> operator*(MyArray<T, SIZE> lhs, const T rhs) {
      lhs *= rhs;
      return lhs;
   }

   __CUDA_HD__
   friend inline MyArray<T, SIZE> operator*(const double lhs, MyArray<T, SIZE> rhs) {
      rhs *= lhs;
      return rhs;
   }

   __CUDA_HD__
   inline MyArray<T, SIZE>& operator*=(const MyArray<T, SIZE> &rhs) {
      __UNROLL__
      for (int i=0; i<SIZE; i++)
         (*this)[i] *= rhs[i];
      return *this;
   }

   __CUDA_HD__
   friend inline MyArray<T, SIZE> operator*(MyArray<T, SIZE> lhs, const  MyArray<T, SIZE> &rhs) {
      lhs *= rhs;
      return lhs;
   }

   __CUDA_HD__
   inline T mod2()       {
      T r = 0.0;
      __UNROLL__
      for (int i=0; i<SIZE; i++)
         r += v[i]*v[i];
      return r;
   }

   __CUDA_HD__
   inline T mod2() const {
      T r = 0.0;
      __UNROLL__
      for (int i=0; i<SIZE; i++)
         r += v[i]*v[i];
      return r;
   }

   __CUDA_HD__
   inline T dot(const  MyArray<T, SIZE> &rhs) const {
      T r = 0.0;
      __UNROLL__
      for (int i=0; i<SIZE; i++)
         r += (*this)[i]*rhs[i];
      return r;
   }

private:
   T v[SIZE];
};

template<typename T, int SIZE_I, int SIZE_J>
struct MyMatrix {
public:
   __CUDA_HD__
   inline void init(const T in) {
      for (int i=0; i<SIZE_I; i++)
         __UNROLL__
         for (int j=0; j<SIZE_J; j++)
         (*this)(i, j) = in;
   }

   __CUDA_HD__
   inline T& operator()(const int i, const int j)       {
#ifdef LEGION_BOUNDS_CHECKS
      assert(i >=     0);
      assert(i < SIZE_I);
      assert(j >=     0);
      assert(j < SIZE_J);
#endif
      return v[i + j*SIZE_I];
   }

   __CUDA_HD__
   inline T  operator()(const int i, const int j) const {
#ifdef LEGION_BOUNDS_CHECKS
      assert(i >=     0);
      assert(i < SIZE_I);
      assert(j >=     0);
      assert(j < SIZE_J);
#endif
      return v[i + j*SIZE_I];
   }

   __CUDA_HD__
   inline void operator=(const MyMatrix<T, SIZE_I, SIZE_J> &in) {
      for (int i=0; i<SIZE_I; i++)
         __UNROLL__
         for (int j=0; j<SIZE_J; j++)
            (*this)(i, j) = in(i, j);
   }

   __CUDA_HD__
   friend inline MyArray<T, SIZE_I> operator*(const MyMatrix<T, SIZE_I, SIZE_J> &lhs, const MyArray<T, SIZE_J> &rhs) {
      MyArray<T, SIZE_I> r;
      // NOTE: This outer unroll increases the compile time a lot the compile time for n O(100)
      //__UNROLL__
      for (int i=0; i<SIZE_I; i++) {
         r[i] = 0.0;
         __UNROLL__
         for (int j=0; j<SIZE_J; j++)
            r[i] += lhs(i,j)*rhs[j];
      }
      return r;
   }

private:
   T v[SIZE_I*SIZE_J];
};

template<typename T, int SIZE>
struct MySymMatrix {
public:
   __CUDA_HD__
   inline T& operator()(const int i, const int j)       {
#ifdef LEGION_BOUNDS_CHECKS
      assert(i >=   0);
      assert(i < SIZE);
      assert(j >=   0);
      assert(j < SIZE);
#endif
      return (i <= j) ? v[(2*SIZE - i - 1)*i/2 + j]
                      : v[(2*SIZE - j - 1)*j/2 + i];
   }

   __CUDA_HD__
   inline T  operator()(const int i, const int j) const {
#ifdef LEGION_BOUNDS_CHECKS
      assert(i >=   0);
      assert(i < SIZE);
      assert(j >=   0);
      assert(j < SIZE);
#endif
      return (i <= j) ? v[(2*SIZE - i - 1)*i/2 + j]
                      : v[(2*SIZE - j - 1)*j/2 + i];
   }

   __CUDA_HD__
   inline MySymMatrix<T, SIZE>& operator=(const MySymMatrix<T, SIZE> &in) {
      __UNROLL__
      for (int i=0; i<SIZE; i++)
         __UNROLL__
         for (int j=i; j<SIZE; j++)
            (*this)(i, j) = in(i, j);
      return *this;
   }

   __CUDA_HD__
   inline MySymMatrix<T, SIZE>& operator+=(const MySymMatrix<T, SIZE> &rhs) {
      __UNROLL__
      for (int i=0; i<SIZE; i++)
         __UNROLL__
         for (int j=i; j<SIZE; j++)
            (*this)(i, j) += rhs(i, j);
      return *this;
   }

   __CUDA_HD__
   inline MySymMatrix<T, SIZE>& operator*=(const double rhs) {
      __UNROLL__
      for (int i=0; i<(SIZE*(SIZE+1)/2); i++)
         v[i] *= rhs;
      return *this;
   }

   __CUDA_HD__
   friend inline MySymMatrix<T, SIZE> operator*(MySymMatrix<T, SIZE> in, const double rhs) {
      in *= rhs;
      return in;
   }

   __CUDA_HD__
   friend inline MySymMatrix<T, SIZE> operator*(const double rhs, MySymMatrix<T, SIZE> in) {
      in *= rhs;
      return in;
   }

   __CUDA_HD__
   inline MySymMatrix<T, SIZE>& operator*=(const MySymMatrix<T, SIZE> &rhs) {
      __UNROLL__
      for (int i=0; i<SIZE; i++)
         (*this)[i] *= rhs[i];
      return *this;
   }

   __CUDA_HD__
   friend inline MySymMatrix<T, SIZE> operator*(MySymMatrix<T, SIZE> in, const MySymMatrix<T, SIZE> &rhs) {
      in *= rhs;
      return in;
   }

private:
   T v[SIZE*(SIZE+1)/2];
};

#endif // __MY_ARRAY_HPP__

