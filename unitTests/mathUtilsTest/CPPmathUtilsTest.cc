
#include <iostream>
#include <assert.h>
#include <math.h>
#include "math_utils.hpp"

void testLUdec() {
   MyMatrix<double, 3, 3> A;
   A(0, 0) = 4.0;
   A(0, 1) = 3.0;
   A(0, 2) = 0.0;
   A(1, 0) = 3.0;
   A(1, 1) = 4.0;
   A(1, 2) =-1.0;
   A(2, 0) = 0.0;
   A(2, 1) =-1.0;
   A(2, 2) = 4.0;
   LUdec<3> LU;
   LU.ludcmp(A);

   MyArray<double, 3> b;
   b[0] = 24.0;
   b[1] = 30.0;
   b[2] =-24.0;
   LU.lubksb(b);

   assert(fabs(b[0] - 3.0) < 1e-12);
   assert(fabs(b[1] - 4.0) < 1e-12);
   assert(fabs(b[2] + 5.0) < 1e-12);
}

// My implicit problem
class myProb : public Rosenbrock<3> {
   inline void rhs(MyArray<double, 3> &r, const MyArray<double, 3> &x) {
      r[0] = -0.013* x[0] - 1000.0*x[0]*x[2];
      r[1] = -2500.0*x[1]*x[2];
      r[2] = -0.013* x[0] - 1000.0*x[0]*x[2] - 2500.0*x[1]*x[2];
   };
};

void testRosenbrock() {
   myProb *p = new myProb;
   MyArray<double, 3> x;
   x[0] = 1.0;
   x[1] = 1.0;
   x[2] = 0.0;

   p->solve(x, 1.0e-3, 25.0);

   assert(fabs(1.0 - (x[0]/( 7.818640e-01))) < 1e-6);
   assert(fabs(1.0 - (x[1]/( 1.218133e+00))) < 1e-6);
   assert(fabs(1.0 - (x[2]/(-2.655799e-06))) < 1e-6);

   delete p;
}

int main() {

   // LU decomposition
   testLUdec();

   // Rosenbrock implicit solver
   testRosenbrock();

   std::cout << "CPPmathUtilsTest: TEST OK!" << std::endl;
   return 0;
}
