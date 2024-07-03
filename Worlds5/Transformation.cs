using System;
using Model;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace Worlds5
{
    public class Transformation
    {
        // Total number of dimensions used
        private static int DimTotal = 5;

        // Matrix containing a single transformation
        private static float[,] manip = new float[DimTotal + 1, DimTotal];
        private static clsSphere.Settings sphereSettings = Model.Globals.Sphere.settings;

        /// <summary>
        /// Initialise the manipulation matrix.
        /// </summary>
        public static void ManipInit()
        {
            Matrix<float> manipMatrix = Matrix<float>.Build.Dense(6, 5, (i, j) => i == j ? 1 : 0);
            manip = manipMatrix.ToArray();
        }

        public static Matrix<float> SetRotation(int axis1, int axis2, float angle)
        {
            // Set the manipulation matrix to a 5x5 identity
            Matrix<float> rotationMatrix = Matrix<float>.Build.DenseIdentity(DimTotal);

            int a = Math.Min(axis1, axis2);
            int b = Math.Max(axis1, axis2);

            double cosAngle = Math.Cos(angle);
            double sinAngle = Math.Sin(angle);

            rotationMatrix[a, a] = (float)cosAngle;
            rotationMatrix[b, b] = (float)cosAngle;
            rotationMatrix[b, a] = (float)sinAngle;
            rotationMatrix[a, b] = -(float)sinAngle;

/*            // Adjust sign for certain higher-dimensional rotations
            if (((b - a) == 2) || ((b - a) == 4))
            {
                rotationMatrix[b, a] = -rotationMatrix[b, a];
                rotationMatrix[a, b] = -rotationMatrix[a, b];
            }*/
            return rotationMatrix;
        }

        public static void SetTranslation(float[] Position)
        {
            ManipInit();
            for (int col = 0; col < DimTotal; ++col)
            {
                manip[DimTotal, col] = Position[col];
            }
        }

        // Matrix pre-multiply for rotations
        public static float[,] PreMulR(Matrix<float> posMatrix, Matrix<float> manipMatrix)
        {
            // Perform matrix multiplication
            var result = posMatrix * manipMatrix;

            // Convert back to float[,] and return
            return result.ToArray();
        }

        // Matrix pre-multiply for translations
        public static void PreMulT()
        {
            for (int row = 0; row < DimTotal; row++)
            {
                for (int col = 0; col < DimTotal; col++)
                {
                    sphereSettings.PositionMatrix[DimTotal, col] += manip[DimTotal, row] * sphereSettings.PositionMatrix[row, col];
                }
            }
        }

/*        // Matrix post-multiply for translations
        public static void PostMulT()
        {
            for (int col = 0; col < DimTotal; col++)
            {
                for (int row = 0; row < DimTotal; row++)
                {
                    sphereSettings.PositionMatrix[DimTotal, col] += sphereSettings.PositionMatrix[DimTotal, row] * manip[row, col];
                }
            }
        }*/

/*        //	Matrix post-multiply for rotations
        public static void PostMulR()
        {
            float[,] temp = new float[6, 6];

            for (int row = 0; row < DimTotal; row++)
            {
                for (int col = 0; col < DimTotal; col++)
                {
                    temp[row, col] = 0;

                    for (int count = 0; count < DimTotal; ++count)
                    {
                        temp[row, col] += sphereSettings.PositionMatrix[row, count] * manip[count, col];
                    }
                }
            }
            for (int row = 0; row < DimTotal; row++)
            {
                for (int col = 0; col < DimTotal; col++)
                {
                    sphereSettings.PositionMatrix[row, col] = temp[row, col];
                }
            }
        }*/

        public static float[,] GetPositionMatrix()
        {
            return sphereSettings.PositionMatrix;
        }

        public static void RotateSphere(RotationCentre rotationCentre, float[,] angles)
        {
            if (rotationCentre == RotationCentre.Origin)
            {
                //float[,] invertedMatrix = Invert6x5Matrix(sphereSettings.PositionMatrix);
                sphereSettings.PositionMatrix = rotateByAngle(sphereSettings.PositionMatrix, angles, true);
            }
            else
            {
                sphereSettings.PositionMatrix = rotateByAngle(sphereSettings.PositionMatrix, angles, false);
            }
        }

        private static float[,] rotateByAngle(float[,] positionMatrix, float[,] angles, bool inverse)
        {
            Matrix<float> posMatrix = Matrix<float>.Build.DenseOfArray(positionMatrix);

            for (int axis1 = 1; axis1 < DimTotal; axis1++)
            {
                for (int axis2 = 2; axis2 <= DimTotal; axis2++)
                {
                    // If rotation is set for this plane
                    if (angles[axis1, axis2] != 0)
                    {
                        // Rotate by the given angle
                        Matrix<float> rotationMatrix = SetRotation(axis1 - 1, axis2 - 1, angles[axis1, axis2]);
                        //PreMulR(posMatrix, manipMatrix);

                        if (inverse)
                        {
                            rotationMatrix = rotationMatrix.Inverse();
                        }
                        posMatrix = posMatrix * rotationMatrix;
                    }
                }
            }
            return posMatrix.ToArray();
        }

        public static Vector5 ImageToFractalSpace(float distance, Vector3 vector3D)
        {
            // Determine the x,y,z coord for this point
            float XPos = distance * vector3D.X;
            float YPos = distance * vector3D.Y;
            float ZPos = distance * vector3D.Z;

            // Transform 3D point x,y,z into nD fractal space at point c[]
            return VectorTrans(XPos, YPos, ZPos);
        }

        public static Vector3 FractalToImageSpace()
        {
            // TODO: Transform nD fractal space at point c[] into 3D point
            return new Vector3(0,0,0);
        }

        public static Vector5 VectorTrans(float x, float y, float z)
        {
            float[] c = new float[DimTotal];
            float[,] matrix = sphereSettings.PositionMatrix;
            for (int i = 0; i < DimTotal; i++)
            {
                c[i] = matrix[0, i] * x +       // Transforms 3D image space at point x,y,z
                       matrix[1, i] * y +       // into nD vector space at point c[]
                       matrix[2, i] * z +
                       matrix[5, i];
            }
            return new Vector5(c[0], c[1], c[2], c[3], c[4]) ;
        }

        static float[,] Invert6x5Matrix(float[,] inputArray)
        {
            // Convert input array to Matrix<float>
            var matrix = Matrix<float>.Build.DenseOfArray(inputArray);

            // Create a 6x6 matrix by adding a column
            var squareMatrix = Matrix<float>.Build.Dense(6, 6);
            squareMatrix.SetSubMatrix(0, 0, matrix);
            squareMatrix.SetColumn(5, Vector<float>.Build.DenseOfArray(new float[] { 0, 0, 0, 0, 0, 1 }));

            // Invert the 6x6 matrix
            var inverted6x6 = squareMatrix.Inverse();

            // Extract the 6x5 portion and convert back to float[,]
            return inverted6x6.SubMatrix(0, 6, 0, 5).ToArray();
        }
    }
}
