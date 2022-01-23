using Model;
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
    public class Helpers
    {
        public static TracedRay.RayDataType[,] ConvertToRayMap(byte[] byteData)
        {
            IntPtr ptr = Marshal.AllocHGlobal(byteData.Length);
            Marshal.Copy(byteData, 0, ptr, byteData.Length);

            var rayType = typeof(TracedRay.RayDataType[,]);
            object rayPtr = Marshal.PtrToStructure(ptr, rayType);
            TracedRay.RayDataType[,] x = (TracedRay.RayDataType[,])rayPtr;
            Marshal.FreeHGlobal(ptr);

            return x;
        }

        public static byte[] Compress(byte[] data)
        {
            MemoryStream output = new MemoryStream();
            using (DeflateStream dstream = new DeflateStream(output, CompressionLevel.Optimal))
            {
                dstream.Write(data, 0, data.Length);
            }
            return output.ToArray();
        }

        public static byte[] Decompress(byte[] data)
        {
            MemoryStream input = new MemoryStream(data);
            MemoryStream output = new MemoryStream();
            using (DeflateStream dstream = new DeflateStream(input, CompressionMode.Decompress))
            {
                dstream.CopyTo(output);
            }
            return output.ToArray();
        }
    }
}
