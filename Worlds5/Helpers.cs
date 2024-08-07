﻿using Model;
using System;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
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

            Array.Copy(intermediate.ExternalPoints, result.ExternalPoints, intermediate.ArraySize);
            Array.Copy(intermediate.ModulusValues, result.ModulusValues, intermediate.ArraySize);
            Array.Copy(intermediate.AngleValues, result.AngleValues, intermediate.ArraySize);
            Array.Copy(intermediate.DistanceValues, result.DistanceValues, intermediate.ArraySize);

            return result;
        }
    }
}
