using System;
using System.Collections;
using System.Data;
using System.Drawing;
using System.Diagnostics;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using Model;

namespace Worlds5 
{
	sealed public class Globals 
	{ 
        public const int TOTAL_THREADS = 8;

		public const int DIB_RGB_COLORS = 0; 
		public const int DIB_PAL_COLORS = 1; 
        //public const int vbSrcCopy = 13369376;
        public const double DEG_TO_RAD = 0.0174532925;

		// ----------------------------------------
		//  Worlds5 Global Constants
		// ----------------------------------------
        
		public const int JPG_FORMAT = 0; 
		public const int BMP_FORMAT = 1; 
		public static long	MaxCount;

		public const float inv60 = (float)0.01667;
		public const float LOG2 = (float)0.69314718;
		public const long MAX_DWORD = 4294967295;
		public const float MAX_FLOAT = (float)3.4E38;

		public static string ReleaseNumber; //  Major, minor & revision
        
		public static byte bRegion; //  Interior/exterior region of fractal selected
		public static RGBTRIPLE[] RGBValues; //  Array to hold true colour bitmap (3 bytes per pixel)
		public static Bitmap bmpImage;	// Bitmap object to hold true colour image
		public static bool UpdatingBitmap = false;

		// ------------------------------------------------------
		//  Type Structure Declarations
		// ------------------------------------------------------
        
		public struct SetUpType 
		{
            public int      MainWidth;
            public int      MainHeight;
            public int      MainLeft;
            public int      MainTop;
            public string   MainState;
            public bool     Toolbar; 
			public bool     Labels; 
			public bool     StatusBar; 
			public bool     ToolTips;
			public string   Sequence; 
			public bool     AutoRepeat; 
			public int      Quality;
            public string   NavPath; 
			public string   SeqPath;
            public int      FramesPerSec;
            public int      BitmapWidth;
            public int      BitmapHeight;
        } 
        
		public struct POINTL 
		{ 
			public int X; 
			public int Y; 
		}

        public struct BITMAPFILEHEADER 
		{ // 14 bytes
			public int bfType; 
			public int bfSize; 
			public int bfReserved1; 
			public int bfReserved2; 
			public int bfOffBits; 
		} 
        
		public struct BITMAPINFOHEADER 
		{ // 40 bytes
			public int biSize; 
			public int biWidth; 
			public int biHeight; 
			public int biPlanes; 
			public int biBitCount; 
			public int biCompression; 
			public int biSizeImage; 
			public int biXPelsPerMeter; 
			public int biYPelsPerMeter; 
			public int biClrUsed; 
			public int biClrImportant; 
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
			public byte rgbBlue; 
			public byte rgbGreen; 
			public byte rgbRed; 
		}

		public struct BITMAPINFO_8 
		{ 
			public BITMAPINFOHEADER bmiHeader; 
			public RGBQUAD[] bmiColors; 
            
			public void Initialize() 
			{ 
				bmiColors = new RGBQUAD[256]; 
			} 
		} 
        
		public struct BITMAPINFO_24 
		{ 
			public BITMAPINFOHEADER bmiHeader; 
			public RGBQUAD bmiColors; 
		} 
        
		public static SetUpType SetUp; 
		public static BITMAPFILEHEADER FileHeader; 
		public static BITMAPINFOHEADER InfoHeader;
        public static BITMAPINFO_24 BitmapInfo24;
    } 
}
