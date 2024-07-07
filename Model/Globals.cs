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
    public enum RotationCentre
    {
        Sphere,
        Origin,
        Trace
    }

    public struct Vector3
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }

        public Vector3(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public static Vector3 operator +(Vector3 v1, Vector3 v2)
        {
            return new Vector3(v1.X + v2.X, v1.Y + v2.Y, v1.Z + v2.Z);
        }

        public static Vector3 operator *(float scalar, Vector3 v)
        {
            return new Vector3(scalar * v.X, scalar * v.Y, scalar * v.Z);
        }

        public static float Dot(Vector3 a, Vector3 b)
        {
            return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        }

        public Vector3 Normalize()
        {
            float magnitude = (float)Math.Sqrt(X * X + Y * Y + Z * Z);
            if (magnitude > 0)
            {
                return new Vector3(X / magnitude, Y / magnitude, Z / magnitude);
            }
            return this;
        }
    }

    public struct Vector5
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
        public float W { get; set; }
        public float V { get; set; }

        public Vector5(float x, float y, float z, float w, float v)
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
        public static Vector5 operator *(float scalar, Vector5 v)
        {
            return new Vector5(scalar * v.X, scalar * v.Y, scalar * v.Z, scalar * v.W, scalar * v.V);
        }

        // The dot product between two 5D vectors
        public static float Dot(Vector5 v1, Vector5 v2)
        {
            return v1.X * v2.X + v1.Y * v2.Y + v1.Z * v2.Z + v1.W * v2.W + v1.V * v2.V;
        }

        public Vector5 Normalize()
        {
            float magnitude = (float)Math.Sqrt(X * X + Y * Y + Z * Z + W * W + V * V);
            if (magnitude > 0)
            {
                return new Vector5(X / magnitude, Y / magnitude, Z / magnitude, W / magnitude, V / magnitude);
            }
            return this;
        }
        public float Magnitude()
        {
            return (float)Math.Sqrt(X * X + Y * Y + Z * Z + W * W + V * V);
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
        public const float DEG_TO_RAD = 0.0174532925f;

        public struct PixelData
        {
            public byte blue;
            public byte green;
            public byte red;
        }

        public struct RGB_QUAD
        {
            public byte rgbBlue;
            public byte rgbGreen;
            public byte rgbRed;
            public byte rgbReserved;
        }

        public struct RGB_TRIPLE
        {
            public int rgbBlue;
            public int rgbGreen;
            public int rgbRed;
        }
	}
}
