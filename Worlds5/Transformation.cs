﻿using Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Worlds5
{
    public class Transformation
    {
        // Matrix containing a single transformation
        private static double[,] manip = new double[6, 6];
        // Total number of dimensions used
        private static int DimTotal = 5;
        private static clsSphere.Settings sphereSettings = Model.Globals.Sphere.settings;

        /// <summary>
        /// Initialise the manipulation matrix.
        /// </summary>
        public static void ManipInit()
        {
            int row, col;

            for (row = 0; row <= 5; ++row)
            {
                for (col = 0; col <= 5; ++col)
                {
                    manip[row, col] = 0;
                    manip[row, col] = 0;
                }
                manip[row, row] = 1;
            }
        }

        public static void SetRotation(int Axis1, int Axis2, double Angle)
        {
            ManipInit();

            int a, b;

            if (Axis1 < Axis2)
            {
                a = Axis1;
                b = Axis2;
            }
            else
            {
                a = Axis2;
                b = Axis1;
            }

            manip[a, a] = Math.Cos(Angle);
            manip[b, b] = Math.Cos(Angle);
            manip[b, a] = Math.Sin(Angle);
            manip[a, b] = -Math.Sin(Angle);

            if (((b - a) == 2) || ((b - a) == 4))
            {
                manip[b, a] = -manip[b, a];
                manip[a, b] = -manip[a, b];
            }
        }

        public static void SetTranslation(double[] Position)
        {
            ManipInit();
            for (int col = 0; col < DimTotal; ++col)
            {
                manip[DimTotal, col] = Position[col];
            }
        }

        // Matrix pre-multiply for rotations
        public static void PreMulR()
        {
            double[,] temp = new double[6, 6];

            for (int row = 0; row < DimTotal; row++)
            {
                for (int col = 0; col < DimTotal; ++col)
                {
                    temp[row, col] = 0;

                    for (int count = 0; count < DimTotal; count++)
                    {
                        temp[row, col] += manip[row, count] * sphereSettings.PositionMatrix[count, col];
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
        }

        // Matrix pre-multiply for translations
        public static void PreMulT()
        {
            for (int col = 0; col < DimTotal; col++)
            {
                for (int row = 0; row < DimTotal; row++)
                {
                    sphereSettings.PositionMatrix[DimTotal, col] += manip[DimTotal, row] * sphereSettings.PositionMatrix[row, col];
                }
            }
        }

        // Matrix post-multiply for translations
        public static void PostMulT()
        {
            for (int col = 0; col < DimTotal; col++)
            {
                for (int row = 0; row < DimTotal; row++)
                {
                    sphereSettings.PositionMatrix[DimTotal, col] += sphereSettings.PositionMatrix[DimTotal, row] * manip[row, col];
                }
            }
        }

        //	Matrix post-multiply for rotations
        public static void PostMulR()
        {
            double[,] temp = new double[6, 6];

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
        }

        public static double[,] GetPositionMatrix()
        {
            return sphereSettings.PositionMatrix;
        }

        public static void RotateSphere(int rotationCentre, double[,] angles)
        {
            double[] Position = new double[DimTotal];

            if (rotationCentre == 1)
            {
                var settings = sphereSettings.Clone();

                //  Invert the rotation
                for (int col = 0; col < DimTotal; ++col)
                {
                    for (int row = 0; row < DimTotal; ++row)
                    {
                        if (row != col)
                        {
                            sphereSettings.PositionMatrix[row, col] = -sphereSettings.PositionMatrix[row, col];
                        }
                    }
                    manip[DimTotal, col] = sphereSettings.PositionMatrix[DimTotal, col];
                    sphereSettings.PositionMatrix[DimTotal, col] = 0;
                }

                PreMulT();

                //  Extract the translation parameters
                for (int col = 0; col < DimTotal; ++col)
                {
                    Position[col] = sphereSettings.PositionMatrix[DimTotal, col];
                }

                sphereSettings = settings.Clone();

                // Translate to the centre coords
                for (int col = 0; col < DimTotal; col++)
                {
                    sphereSettings.PositionMatrix[DimTotal, col] = 0;
                }
            }

            for (int axis1 = 1; axis1 < DimTotal; axis1++)
            {
                for (int axis2 = 2; axis2 <= DimTotal; axis2++)
                {
                    // If rotation is set for this plane
                    if (angles[axis1, axis2] != 0)
                    {
                        // Rotate by the given angle
                        SetRotation(axis1 - 1, axis2 - 1, angles[axis1, axis2]);
                        PreMulR();
                    }
                }
            }

            if (rotationCentre == 1)
            {
                // Translate back from the centre
                SetTranslation(Position);
                PreMulT();
            }
        }
    }
}
