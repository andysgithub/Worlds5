using System;
using Model;
using MathNet.Numerics.LinearAlgebra;

namespace Worlds5
{
    public class Transformation
    {
        // Total number of dimensions used
        private static int DimTotal = 5;
        // Record the position matrix in the sphere
        private static float[,] positionMatrix = Model.Globals.Sphere.settings.PositionMatrix;

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

            return rotationMatrix;
        }

        public static void RotateImage(RotationCentre rotationCentre, float[,] angles)
        {
            Matrix<float> posMatrix = Matrix<float>.Build.DenseOfArray(positionMatrix);
            Matrix<float> rotationMatrix = CalculateRotationMatrix(angles);

            if (rotationCentre == RotationCentre.Origin)
            {
                // Rotate fractal space (inverse rotation)
                posMatrix = posMatrix * rotationMatrix.Inverse();
            }
            else
            {
                // Rotate image space
                posMatrix = ApplyRotationToPositionMatrix(posMatrix, rotationMatrix);
            }

            positionMatrix = posMatrix.ToArray();
        }

        private static Matrix<float> CalculateRotationMatrix(float[,] angles)
        {
            Matrix<float> rotationMatrix = Matrix<float>.Build.DenseIdentity(DimTotal);
            for (int axis1 = 1; axis1 < DimTotal; axis1++)
            {
                for (int axis2 = 2; axis2 <= DimTotal; axis2++)
                {
                    // If rotation is set for this plane
                    if (angles[axis1, axis2] != 0)
                    {
                        // Rotate by the given angle
                        Matrix<float> planeRotation = SetRotation(axis1 - 1, axis2 - 1, angles[axis1, axis2]);
                        rotationMatrix = rotationMatrix * planeRotation;
                    }
                }
            }
            return rotationMatrix;
        }

        private static Matrix<float> ApplyRotationToPositionMatrix(Matrix<float> posMatrix, Matrix<float> rotationMatrix)
        {
            // Create a new 6x5 matrix for the result
            var result = Matrix<float>.Build.Dense(6, 5);

            // Apply rotation to the first 5 rows (the linear transformation part)
            var linearPart = posMatrix.SubMatrix(0, 5, 0, 5);
            var rotatedLinearPart = rotationMatrix * linearPart;
            result.SetSubMatrix(0, 0, rotatedLinearPart);

            // Copy the last row (translation) as is
            result.SetRow(5, posMatrix.Row(5));

            return result;
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

        public static Vector5 VectorTrans(float x, float y, float z)
        {
            float[] c = new float[DimTotal];
            float[,] matrix = positionMatrix;
            for (int i = 0; i < DimTotal; i++)
            {
                c[i] = matrix[0, i] * x +       // Transforms 3D image space at point x,y,z
                       matrix[1, i] * y +       // into nD vector space at point c[]
                       matrix[2, i] * z +
                       matrix[5, i];
            }
            return new Vector5(c[0], c[1], c[2], c[3], c[4]) ;
        }

        public static float[,] GetPositionMatrix()
        {
            return positionMatrix;
        }
    }
}
