using Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using static Model.TracedRay;

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

        public static RayDataType ConvertFromIntermediate(RayDataTypeIntermediate intermediate)
        {
            RayDataType result = new RayDataType
            {
                BoundaryTotal = intermediate.BoundaryTotal,
                ExternalPoints = new int[intermediate.ArraySize],
                ModulusValues = new float[intermediate.ArraySize],
                AngleValues = new float[intermediate.ArraySize],
                DistanceValues = new float[intermediate.ArraySize]
            };

            Marshal.Copy(intermediate.ExternalPoints, result.ExternalPoints, 0, intermediate.ArraySize);
            Marshal.Copy(intermediate.ModulusValues, result.ModulusValues, 0, intermediate.ArraySize);
            Marshal.Copy(intermediate.AngleValues, result.AngleValues, 0, intermediate.ArraySize);
            Marshal.Copy(intermediate.DistanceValues, result.DistanceValues, 0, intermediate.ArraySize);

            return result;
        }
    }
}
