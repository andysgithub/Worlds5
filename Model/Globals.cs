using System;

namespace Model
{
	/// <summary>
	/// Global structures for Worlds5.
	/// </summary>
	public class Globals
	{
		public static int Dimensions = 5;	//  Total dimensions for the current location
        public const byte MaxColours = 255;
        public static clsSphere Sphere = null;
        public const double DEG_TO_RAD = 0.0174532925;

        //public struct TransMatrixType 
        //{  
        //    public float[,] HSL; 
        //    public float[] ColourDetail; 
            
        //    public void Initialize() 
        //    {  
        //        HSL = new float[ 3, 2 ]; 
        //        ColourDetail = new float[ 2 ]; 
        //    } 
        //}

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
