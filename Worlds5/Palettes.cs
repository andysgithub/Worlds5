using System;
using System.IO;
using System.Collections;
using System.Data;
using System.Drawing;
using System.Diagnostics;
using System.Windows.Forms;
using System.Reflection;
using System.Runtime.InteropServices;
using DataClasses;

namespace Worlds5 
{
	sealed public class Palettes 
	{ 
		//  Author:       Andrew G Williams
		// 
		//  Compiler:     Visual C#
		//  System:       Windows XP
		// 
		//  Module name:  Bitmaps
		// 
		//  This module defines data structures for 8-bit & 24-bit bitmaps
		//  and loads the image section of a bitmap into an array.
		// 
		//  Full bitmap file consists of the following structures::
		//  FileHeader: General file header specifying bitmap type and size
		//  InfoHeader: Specific bitmap info (height, width etc)
		//  Palette (8-bit only): 256-colour palette, 4 bytes per colour (B,G,R,<reserved>)
		//  PaletteValues(): Bitmap image data
		// 
        
		//  Palette function declarations
        [DllImport("gdi32.dll", EntryPoint = "CreatePalette")]
        private static extern int CreatePalette(ref Globals.LOGPALETTE LPalette);
        [DllImport("gdi32.dll", EntryPoint = "SelectPalette")]
        private static extern int SelectPalette(int hdc, int hPalette, int bForceBackground);
        [DllImport("gdi32.dll", EntryPoint = "RealizePalette")]
        private static extern int RealizePalette(int hdc);
        
		public static void InitPalettes()
		{
			DataClasses.Globals.Palette8[0].Initialize();
			DataClasses.Globals.Palette8[1].Initialize();
		}

		///  Loads an individual palette or generates a merged palette.
		public static void LoadPalette( string Palette ) 
		{ 
			int PlusPos = 0;			//  Position of the "+" symbol in a merged palette string
			int MulPos = 0;				//  Position of the "*" symbol in a merged palette string
			string Palette1 = null;		//  The first palette name in a merged palette string
			string Palette2 = null;		//  The second palette name in a merged palette string
			double Proportion = 0;		//  Proportion (0-1) of the second palette in a merged palette
            
			PlusPos = Palette.IndexOf("+"); 
			if ( PlusPos != -1 ) 
			{ 
				//  Generate the merged palette if this frame has two palettes combined
				MulPos = Palette.IndexOf("*");
				Palette1 = Palette.Substring(0, PlusPos);
                Palette2 = Palette.Substring(PlusPos + 1, MulPos - PlusPos - 1);
                Proportion = double.Parse(Palette.Substring(Palette.Length - MulPos));
                CombinePalettes(Palette1, Palette2, Proportion); 
			} 
			else 
			{ 
				LoadSinglePalette( ref Palette, ref DataClasses.Globals.Palette8[Globals.bRegion] ); 
			} 
		} 
        
		///  Load an individual palette file.
		///  Checks to ensure that the palette is not a merged palette.
		///  Changes the palette name to the most prominent palette otherwise.
        public static void LoadSinglePalette(ref string Palette, ref DataClasses.Globals.PALETTE_8 Palette8) 
		{ 
			int PlusPos;		//  Position of the "+" symbol in a merged palette string
			int MulPos;			//  Position of the "*" symbol in a merged palette string
			double Proportion;	//  Proportion (0-1) of the second palette in a merged palette
            
			try 
			{ 
				byte[] temp = new byte[1080];
               
				//  Read the file header
                string FileName = Path.Combine(Globals.SetUp.PalPath, Palette + ".bmp");

                FileStream fs = new FileStream(FileName, FileMode.Open, FileAccess.Read, FileShare.None, 1024);

				//  Read the 8-bit BITMAPINFO structure
                int count = fs.Read(temp, 0, 1079);

				int Index;
				for(int i=0; i<256; i++)
				{
					Index = i*4;
                    Palette8.bmiColors[i].rgbBlue = temp[Index + 54];
                    Palette8.bmiColors[i].rgbGreen = temp[Index + 55];
                    Palette8.bmiColors[i].rgbRed = temp[Index + 56];
                    Palette8.bmiColors[i].rgbReserved = temp[Index + 57];
				}
                
				fs.Close();
				return; 
			} 
			catch (Exception ex)
			{ 
				Console.WriteLine(ex);
				MessageBox.Show(ex.Message, "LoadSinglePalette1", MessageBoxButtons.OK, MessageBoxIcon.Exclamation); 
			} 
            
			//  Check to see if this is a merged palette
			PlusPos = Palette.IndexOf("+"); 

			if ( PlusPos >= 0 ) 
			{ 
				//  If so, find which palette is the more prominent
                MulPos = Palette.IndexOf("*");
                Proportion = Convert.ToDouble(Palette.Substring(MulPos + 1, Palette.Length - MulPos - 1));

                if (Proportion < 0.5)
                    //  Use the first palette from the combination
                    Palette = Palette.Substring(0, PlusPos);
                else
                    //  Use the second palette from the combination
                    Palette = Palette.Substring(PlusPos + 1, MulPos - PlusPos);
			} 
			else if (Palette != "sky") 
				//  Not a merged palette, so try loading the default palette
				Palette = "sky"; 
		} 
        
		public static void CombinePalettes(string Palette1, string Palette2, double Proportion) 
		{ 
			//  Produce combination of two palettes according to the Proportion (0-1) of the second palette
            DataClasses.Globals.PALETTE_8 Palette8b;

            Palette8b.bmiColors = null;
            
			//  Read palette1 header data
            DataClasses.Globals.PALETTE_8 transTemp0 = DataClasses.Globals.Palette8[Globals.bRegion];
            LoadSinglePalette(ref Palette1, ref transTemp0); 
            
			//  Read palette2 header data
            LoadSinglePalette(ref Palette2, ref Palette8b);
            
			// '''''''''''''''''''''''''''''''''''''''''''''''''''''
			//    Produce colours from R1,G1,B1 & R2,G2,B2 values
			// '''''''''''''''''''''''''''''''''''''''''''''''''''''
            
			int Count = 0; 
			byte g = 0, r = 0, b = 0; 
			byte G1 = 0, R1 = 0, B1 = 0; 
			byte g2 = 0, r2 = 0, b2 = 0; 
            
			for ( Count=0; Count<=255; Count++ ) 
			{
                R1 = DataClasses.Globals.Palette8[Globals.bRegion].bmiColors[Count].rgbRed;
                G1 = DataClasses.Globals.Palette8[Globals.bRegion].bmiColors[Count].rgbGreen;
                B1 = DataClasses.Globals.Palette8[Globals.bRegion].bmiColors[Count].rgbBlue;
                
				r2 = Palette8b.bmiColors[Count].rgbRed; 
				g2 = Palette8b.bmiColors[Count].rgbGreen; 
				b2 = Palette8b.bmiColors[Count].rgbBlue;

                r = Convert.ToByte(System.Convert.ToDecimal(R1 * (1.0 - Proportion) + r2 * Proportion));
                g = Convert.ToByte(System.Convert.ToDecimal(G1 * (1 - Proportion) + g2 * Proportion));
                b = Convert.ToByte(System.Convert.ToDecimal(B1 * (1 - Proportion) + b2 * Proportion));

                SetBuf(r, g, b, Count); 
			} 
		} 
        
		public static void SetBuf( byte r, byte g, byte b, int Count ) 
		{ 
			//  Store the palette into the BITMAPINFO.bmiColors() array
            DataClasses.Globals.Palette8[Globals.bRegion].bmiColors[Count].rgbRed = r; 
			DataClasses.Globals.Palette8[Globals.bRegion].bmiColors[Count].rgbGreen = g; 
			DataClasses.Globals.Palette8[Globals.bRegion].bmiColors[Count].rgbBlue = b; 
		} 
        
		public static void PreShift() 
		{ 
			DataClasses.Globals.RGBQUAD[] ShiftedPalette = new DataClasses.Globals.RGBQUAD[256]; 
			int i = 0; 
            
			//  Shift colours according to ColourOffset
			for ( i=0; i<=255; i++ ) 
			{
                float fColourVal = i + DataClasses.Globals.Sphere.ColourOffset[Globals.bRegion];
                int iColourIndex = Convert.ToInt32(fColourVal) % 256;
                ShiftedPalette[i] = DataClasses.Globals.Palette8[Globals.bRegion].bmiColors[iColourIndex];
			} 
            
			for (i = 0; i <= 255; i++) 
			{ 
				DataClasses.Globals.Palette8[Globals.bRegion].bmiColors[i] = ShiftedPalette[i]; 
			} 
		} 
	} 
}