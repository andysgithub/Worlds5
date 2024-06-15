using System;
using System.CodeDom;

namespace Model
{
    public enum AxisPair
    {
        XY,
        XZ,
        XW,
        XV,
        YZ,
        YW,
        YV,
        ZW,
        ZV,
        WV
    }

    public struct Vector3
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Z { get; set; }

        public Vector3(double x, double y, double z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public static Vector3 operator +(Vector3 v1, Vector3 v2)
        {
            return new Vector3(v1.X + v2.X, v1.Y + v2.Y, v1.Z + v2.Z);
        }

        public static Vector3 operator *(double scalar, Vector3 v)
        {
            return new Vector3(scalar * v.X, scalar * v.Y, scalar * v.Z);
        }
        public Vector3 Normalize()
        {
            double magnitude = Math.Sqrt(X * X + Y * Y + Z * Z);
            if (magnitude > 0)
            {
                return new Vector3(X / magnitude, Y / magnitude, Z / magnitude);
            }
            return this;
        }
    }

    public struct Vector5
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Z { get; set; }
        public double W { get; set; }
        public double V { get; set; }

        public Vector5(double x, double y, double z, double w, double v)
        {
            X = x;
            Y = y;
            Z = z;
            W = w;
            V = v;
        }

        // Operator to add two 5D vectors
        public static Vector5 operator +(Vector5 v1, Vector5 v2)
        {
            return new Vector5(v1.X + v2.X, v1.Y + v2.Y, v1.Z + v2.Z, v1.W + v2.W, v1.V + v2.V);
        }

        // Operator to subtract two 5D vectors
        public static Vector5 operator -(Vector5 v1, Vector5 v2)
        {
            return new Vector5(v1.X - v2.X, v1.Y - v2.Y, v1.Z - v2.Z, v1.W - v2.W, v1.V - v2.V);
        }

        // Operator to multiply a 5D vector by a scalar value
        public static Vector5 operator *(double scalar, Vector5 v)
        {
            return new Vector5(scalar * v.X, scalar * v.Y, scalar * v.Z, scalar * v.W, scalar * v.V);
        }

        // The dot product between two 5D vectors
        public static double Dot(Vector5 v1, Vector5 v2)
        {
            return v1.X * v2.X + v1.Y * v2.Y + v1.Z * v2.Z + v1.W * v2.W + v1.V * v2.V;
        }

        public Vector5 Normalize()
        {
            double magnitude = Math.Sqrt(X * X + Y * Y + Z * Z + W * W + V * V);
            if (magnitude > 0)
            {
                return new Vector5(X / magnitude, Y / magnitude, Z / magnitude, W / magnitude, V / magnitude);
            }
            return this;
        }
        public double Magnitude()
        {
            return Math.Sqrt(X * X + Y * Y + Z * Z + W * W + V * V);
        }
    }

    /// <summary>
    /// Global structures for Worlds5.
    /// </summary>
    public class Globals
	{
        public static string AppName = "Worlds 5";
		public static int Dimensions = 5;	//  Total dimensions for the current location
        public static string CurrentAddress = "";
        public static clsSphere Sphere = null;
        public const double DEG_TO_RAD = 0.0174532925;

        public struct PixelData
        {
            public byte blue;
            public byte green;
            public byte red;
        }

        public struct RGBQUAD
        {
            public byte rgbBlue;
            public byte rgbGreen;
            public byte rgbRed;
            public byte rgbReserved;
        }

        public struct RGBTRIPLE
        {
            public int rgbBlue;
            public int rgbGreen;
            public int rgbRed;
        }

        //public struct vertex
        //{
        //    public double X;
        //    public double Y;
        //    public double Z;
        //}
        
	}

    //public class ImagePlane
    //{
    //    public double Left, Right, Top, Bottom, Width, Height;

    //    public ImagePlane(double Left, double Top, double Width, double Height)
    //    {
    //        this.Left = Left;
    //        this.Top = Top;
    //        this.Width = Width;
    //        this.Height = Height;
    //        this.Right = Left + Width;
    //        this.Bottom = Top + Height;
    //    }
    //}
}
